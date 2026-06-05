#!/usr/bin/env python3
"""profile_backward.py — where does the real backward's ~100ms/fire go?

Builds random base weights at real Llama-3.2-1B (or 8B) dims and times the
phases of one backward over a realistic FT prefill (n tokens, one sequence):
  - head_backward (LM-head over full vocab + final-norm)
  - per-layer loop: layer_forward (rematerialize) + layer_backward
  - optimizer.step (fused AdamW)

All on a CUDA stream with torch.cuda.synchronize() around each phase (the
backward is async on its own stream in prod; here we sync to attribute time).
This tells us whether to attack rematerialization, the attention bwd, the
head, or the optimizer — before reaching for MPS isolation.

Usage: python scripts/profile_backward.py [--model 1b|8b] [--n 100] [--iters 10]
"""
from __future__ import annotations
import argparse, os, sys, time
import torch

try:
    from sglang.srt.deltaserve.bwd_services.llama3 import (
        head_backward, layer_forward, layer_backward, rope_cos_sin)
except Exception:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sglang-fork", "sglang", "srt"))
    from deltaserve.bwd_services.llama3 import (  # type: ignore
        head_backward, layer_forward, layer_backward, rope_cos_sin)

CFG = {
    "1b": dict(D=2048, L=16, Hq=32, Hkv=8, Hd=64, inter=8192, vocab=128256),
    "8b": dict(D=4096, L=32, Hq=32, Hkv=8, Hd=128, inter=14336, vocab=128256),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["1b", "8b"], default="1b")
    ap.add_argument("--n", type=int, default=100, help="FT prefill token count")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()
    c = CFG[args.model]
    D, L, Hq, Hkv, Hd = c["D"], c["L"], c["Hq"], c["Hkv"], c["Hd"]
    inter, vocab = c["inter"], c["vocab"]
    kv_size = Hkv * Hd; q_size = Hq * Hd
    dims = (Hq, Hkv, Hd, kv_size)
    dev = "cuda"; dt = torch.bfloat16
    scaling = 32.0 / args.rank; eps = 1e-5; n = args.n
    print(f"[profile] {args.model} D={D} L={L} Hq={Hq} Hkv={Hkv} Hd={Hd} inter={inter} "
          f"vocab={vocab} | n={n} rank={args.rank} dtype={dt}")

    rb = lambda *s: (torch.randn(*s, device=dev, dtype=dt) * 0.02)
    base = [{"q": rb(q_size, D), "k": rb(kv_size, D), "v": rb(kv_size, D), "o": rb(D, D),
             "gate": rb(inter, D), "up": rb(inter, D), "down": rb(D, inter),
             "in_ln": torch.ones(D, device=dev, dtype=dt), "post_ln": torch.ones(D, device=dev, dtype=dt)}
            for _ in range(L)]
    norm_w = torch.ones(D, device=dev, dtype=dt)
    lm_w = rb(vocab, D)
    lora = []
    for _ in range(L):
        ld = {}
        for proj, od in [("q", q_size), ("o", q_size), ("k", kv_size), ("v", kv_size)]:
            ld[proj] = {"A": (torch.randn(args.rank, D, device=dev, dtype=torch.float32) * 0.01),
                        "B": torch.zeros(od, args.rank, device=dev, dtype=torch.float32)}
        lora.append(ld)
    params = [lora[i][p][ab] for i in range(L) for p in ("q", "k", "v", "o") for ab in ("A", "B")]
    opt = torch.optim.AdamW(params, lr=5e-6, fused=True)

    seq_lens = [n]; b_start = [0]
    positions = torch.arange(n, device=dev)
    cos, sin = rope_cos_sin(positions, Hd, 500000.0)
    ids = torch.randint(0, vocab, (n,), device=dev)
    layer_in = {i: rb(n, D) for i in range(L)}
    final_in = rb(n, D)

    def lw_of(i):
        d = dict(base[i])
        for p in ("q", "k", "v", "o"):
            d[p + "A"] = lora[i][p]["A"]; d[p + "B"] = lora[i][p]["B"]
        return d

    def one(timing):
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        loss, n_valid, g = head_backward(final_in, lm_w, norm_w, eps, ids, seq_lens, b_start, vocab)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        t_fwd = t_bwd = 0.0
        for i in range(L - 1, -1, -1):
            lw = lw_of(i)
            torch.cuda.synchronize(); a = time.perf_counter()
            with torch.no_grad():
                cache = layer_forward(layer_in[i], lw, scaling, cos, sin, seq_lens, b_start, dims, eps)
            torch.cuda.synchronize(); b = time.perf_counter()
            grad_x, grads = layer_backward(g, cache, lw, scaling, cos, sin, seq_lens, b_start, dims, eps, cdt=dt)
            for p in ("q", "k", "v", "o"):
                for ab in ("A", "B"):
                    gg = grads[p + ab]
                    pp = lora[i][p][ab]
                    pp.grad = gg.float() if pp.grad is None else pp.grad + gg.float()
            g = grad_x
            torch.cuda.synchronize(); cc = time.perf_counter()
            t_fwd += b - a; t_bwd += cc - b
        torch.cuda.synchronize(); t2 = time.perf_counter()
        opt.step()
        torch.cuda.synchronize(); t3 = time.perf_counter()
        timing["head"] += t1 - t0
        timing["layer_forward(remat)"] += t_fwd
        timing["layer_backward"] += t_bwd
        timing["optimizer"] += t3 - t2
        timing["total"] += t3 - t0

    warm = {k: 0.0 for k in ["head", "layer_forward(remat)", "layer_backward", "optimizer", "total"]}
    one(warm)  # warmup (first-touch alloc/compile)
    timing = {k: 0.0 for k in warm}
    for _ in range(args.iters):
        one(timing)
    print(f"[profile] mean over {args.iters} iters (after 1 warmup):")
    for k in ["head", "layer_forward(remat)", "layer_backward", "optimizer", "total"]:
        ms = timing[k] / args.iters * 1000
        pct = 100 * timing[k] / timing["total"]
        print(f"    {k:24s} {ms:8.2f} ms   {pct:5.1f}%")


if __name__ == "__main__":
    main()
