#!/usr/bin/env python3
"""test_subprocess_backward.py — S12a Milestone 1: validate the backward
subprocess plumbing end-to-end, WITHOUT needing the sglang server.

Builds a synthetic base-weight state (tiny dims), then:
  1. in-process: build_real_backward_from_state(state) → process(snapshot) → loss_ref
  2. subprocess: spawn backward_process child, send {init, backward}, get loss_sub
Asserts loss_sub == loss_ref (the head-CE loss is determined by the snapshot's
final_in, independent of the random LoRA init, so they must match bit-for-bit).

This proves: child boots, torch.loads the /dev/shm state, builds the verified
backward from it, receives an IPC snapshot, runs it on its own GPU, returns the
right loss. MPS capping is orthogonal (add --mps to run the child capped).

Usage: python scripts/test_subprocess_backward.py [--mps 10]
"""
from __future__ import annotations
import argparse, os, pickle, sys, time
import torch
import zmq

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "sglang-fork", "sglang", "srt"))
from deltaserve.real_backward import build_real_backward_from_state  # noqa
from deltaserve.backward_process import spawn_backward_process       # noqa

# tiny synthetic config — plumbing test, not a perf test
C = dict(hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
         num_key_value_heads=4, head_dim=32, intermediate_size=512,
         rope_theta=10000.0, rms_norm_eps=1e-5, vocab_size=1000)
DT = torch.bfloat16


def make_state():
    D, L = C["hidden_size"], C["num_hidden_layers"]
    Hq, Hkv, Hd = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"]
    q_size, kv_size, inter, vocab = Hq * Hd, Hkv * Hd, C["intermediate_size"], C["vocab_size"]
    g = torch.Generator().manual_seed(7)
    rb = lambda *s: (torch.randn(*s, generator=g, dtype=torch.float32) * 0.02).to(DT)
    base = [{
        "q": rb(q_size, D), "k": rb(kv_size, D), "v": rb(kv_size, D), "o": rb(D, D),
        "gate": rb(inter, D), "up": rb(inter, D), "down": rb(D, inter),
        "in_ln": torch.ones(D, dtype=DT), "post_ln": torch.ones(D, dtype=DT),
    } for _ in range(L)]
    return {"config": C, "base": base, "norm_w": torch.ones(D, dtype=DT),
            "lm_w": rb(vocab, D), "base_dtype": "torch.bfloat16"}


def make_snapshot(n=20):
    D, L, vocab = C["hidden_size"], C["num_hidden_layers"], C["vocab_size"]
    g = torch.Generator().manual_seed(11)
    rb = lambda *s: (torch.randn(*s, generator=g, dtype=torch.float32) * 0.1).to(DT)
    return {
        "layer_in": {i: rb(n, D) for i in range(L)},
        "final_in": rb(n, D),
        "concat_input_ids": torch.randint(0, vocab, (n,), generator=g),
    }, [n]


def in_process_ref(state, snap, sample_lens):
    s = {k: v for k, v in state.items()}
    # move to cuda
    s = {"config": s["config"], "base_dtype": s["base_dtype"],
         "norm_w": s["norm_w"], "lm_w": s["lm_w"], "base": s["base"]}
    rb = build_real_backward_from_state(s, device="cuda")
    snap_cuda = {"layer_in": {i: t.cuda() for i, t in snap["layer_in"].items()},
                 "final_in": snap["final_in"].cuda(),
                 "concat_input_ids": snap["concat_input_ids"].cuda()}
    rb.process(snap_cuda, sample_lens=sample_lens)
    torch.cuda.synchronize()
    return rb._cum_loss / max(1, rb._call_count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mps", type=int, default=0, help="MPS thread %% for child (0=off)")
    args = ap.parse_args()

    state = make_state()
    snap, sample_lens = make_snapshot()

    print("[test] in-process reference...")
    loss_ref = in_process_ref(state, snap, sample_lens)
    print(f"[test] loss_ref = {loss_ref:.6f}")

    wpath = f"/dev/shm/ds_test_state_{os.getpid()}.pt"
    torch.save(state, wpath)

    addr = f"ipc:///tmp/ds_test_bwd_{os.getpid()}"
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PAIR)
    sock.connect(addr)

    env = dict(os.environ)
    proc = spawn_backward_process(addr, "synthetic", args.mps, env=env)
    print(f"[test] spawned child pid={proc.pid} mps={args.mps or 'off'}")

    rc = 1
    try:
        # init
        sock.send(pickle.dumps({"op": "init", "weights_path": wpath}))
        if not sock.poll(120000):
            print("[test] FAIL: child init timeout"); return 3
        r = pickle.loads(sock.recv())
        assert r.get("ok"), f"init failed: {r}"
        print(f"[test] child init ok: L={r.get('L')} D={r.get('D')}")

        # backward
        sock.send(pickle.dumps({"op": "backward", "snapshot": snap,
                                "sample_lens": sample_lens, "sync": True}))
        if not sock.poll(60000):
            print("[test] FAIL: child backward timeout"); return 4
        r = pickle.loads(sock.recv())
        loss_sub = r.get("loss")
        print(f"[test] loss_sub = {loss_sub:.6f}  (child {r.get('last_call_ms'):.1f}ms)")

        gap = abs(loss_sub - loss_ref)
        ok = gap < 1e-4
        print(f"[test] |loss_sub - loss_ref| = {gap:.2e}")
        print(f"[test] RESULT: {'PASS' if ok else 'FAIL'} — subprocess backward "
              f"{'matches' if ok else 'DIVERGES FROM'} in-process")
        rc = 0 if ok else 1

        sock.send(pickle.dumps({"op": "shutdown"})); sock.poll(5000)
    finally:
        try: proc.wait(timeout=10)
        except Exception: proc.kill()
        sock.close(linger=0)
        try: os.remove(wpath)
        except OSError: pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
