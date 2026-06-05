#!/usr/bin/env python3
"""mps_probe.py — does MPS thread-% capping protect a latency-sensitive process
from a compute-hog co-tenant on this GPU? (de-risks S12a before building it.)

Two roles on the same GPU:
  --role victim : bursts of "inference prefill"-shaped GEMMs, measures per-burst
                  latency (this is the TTFT-analog we care about). Sleeps between
                  bursts to mimic request arrival.
  --role hog    : continuous "backward"-shaped GEMM sequence (~the real 94ms/fire),
                  fires repeatedly to contend for SMs.

Orchestrated by mps_probe.sh across 3 conditions:
  A victim alone                  -> baseline latency
  B victim + hog, NO MPS daemon   -> default time-slice contention
  C victim + hog, MPS, hog @ 10%  -> isolation test

If C's victim latency ≈ A and ≪ B, MPS honors the cap -> S12a is worth building.
"""
from __future__ import annotations
import argparse, os, sys, time
import torch

D = 4096          # 8B hidden
INTER = 14336
L = 32
DEV = "cuda"
DT = torch.bfloat16


def victim(iters, sleep_ms):
    torch.cuda.init()
    # one "prefill": n=80 tokens through a few projection-sized GEMMs
    n = 80
    x = torch.randn(n, D, device=DEV, dtype=DT)
    wq = torch.randn(D, D, device=DEV, dtype=DT)
    wgu = torch.randn(D, 2 * INTER, device=DEV, dtype=DT)
    wd = torch.randn(INTER, D, device=DEV, dtype=DT)

    def one_prefill():
        h = x
        for _ in range(L):
            q = h @ wq
            gu = h @ wgu
            g, u = gu[:, :INTER], gu[:, INTER:]
            h = (torch.nn.functional.silu(g) * u) @ wd + q[:, :D]
        return h

    # warmup
    for _ in range(5):
        one_prefill()
    torch.cuda.synchronize()

    lats = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        one_prefill()
        torch.cuda.synchronize(); lats.append((time.perf_counter() - t0) * 1000)
        time.sleep(sleep_ms / 1000)
    lats.sort()
    mean = sum(lats) / len(lats)
    p95 = lats[int(len(lats) * 0.95)]
    print(f"VICTIM_RESULT mean={mean:.2f}ms p50={lats[len(lats)//2]:.2f}ms "
          f"p95={p95:.2f}ms n={len(lats)}", flush=True)


def hog(seconds):
    torch.cuda.init()
    # "backward"-shaped: big GEMMs mimicking layer_backward across L layers
    n = 100
    a = torch.randn(n, D, device=DEV, dtype=DT)
    w1 = torch.randn(D, INTER, device=DEV, dtype=DT)
    w2 = torch.randn(INTER, D, device=DEV, dtype=DT)
    wq = torch.randn(D, D, device=DEV, dtype=DT)
    for _ in range(5):
        h = a @ w1; (h @ w2) @ wq
    torch.cuda.synchronize()
    deadline = time.time() + seconds
    fires = 0
    while time.time() < deadline:
        # one "fire" = L layers of backward GEMMs (~the 94ms sequence)
        for _ in range(L):
            h = a @ w1
            g = h @ w2
            _ = g @ wq
        torch.cuda.synchronize()
        fires += 1
    print(f"HOG_RESULT fires={fires} in {seconds}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["victim", "hog"], required=True)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--sleep-ms", type=float, default=50)
    ap.add_argument("--seconds", type=float, default=12)
    args = ap.parse_args()
    if args.role == "victim":
        victim(args.iters, args.sleep_ms)
    else:
        hog(args.seconds)


if __name__ == "__main__":
    main()
