#!/usr/bin/env python3
"""Adapt our auto_benchmark_sglang outputs into the inputs compare_slora_dserve.py
expects.

compare_slora_dserve.py wants:
  - results CSV: idx,t_rel_s,latency_s,status,ttft_s
  - bwd_log CSV: timestamp (ISO), batch_tokens

Our auto_benchmark writes:
  - results: rid,sent_t,ttft_s,latency_s,...,error
  - FT fires: server log lines `real_backward #N: ...ms loss=.. n_valid=K cum_tokens=.. wall=<epoch>`

This converts both. The bwd_log timestamp is the fire's wall-clock (ISO); the
plot script anchors the first fire at the dserve offset, so relative timing is
what matters.
"""
import argparse, csv, re
from datetime import datetime

_FIRE = re.compile(r"real_backward #\d+:.*n_valid=(\d+)\s+cum_tokens=\d+\s+wall=([\d.]+)")


def conv_results(src, dst):
    rows = list(csv.DictReader(open(src)))
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "t_rel_s", "latency_s", "status", "ttft_s"])
        n = 0
        for i, r in enumerate(rows):
            status = "ok" if not (r.get("error") or "").strip() else "error"
            w.writerow([i, r.get("sent_t", ""), r.get("latency_s", ""),
                        status, r.get("ttft_s", "")])
            n += 1
    print(f"[adapt] results {src} -> {dst} ({n} rows)")


def conv_bwd(server_log, dst, anchor_wall=None, tl_span=None):
    """Trim to the timeline phase [anchor_wall, anchor_wall+tl_span] so the FT
    curve aligns to the inference clock (the plot script assumes the bwd_log is
    already trimmed — first row anchors at tl_base). Without anchor_wall, keep all
    fires (first fire = pre-timeline warmup → misaligned)."""
    fires = []
    for line in open(server_log, errors="ignore"):
        m = _FIRE.search(line)
        if m:
            fires.append((float(m.group(2)), int(m.group(1))))   # (wall, n_valid)
    if anchor_wall is not None:
        lo, hi = anchor_wall, (anchor_wall + tl_span if tl_span else float("inf"))
        fires = [(w, n) for (w, n) in fires if lo <= w <= hi]
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "batch_tokens"])
        for wall, ntok in fires:
            w.writerow([datetime.fromtimestamp(wall).isoformat(), ntok])
    print(f"[adapt] bwd {server_log} -> {dst} ({len(fires)} fires, "
          f"{sum(n for _,n in fires)} tok)" +
          (f" [trimmed to timeline {tl_span}s]" if anchor_wall else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-in"); ap.add_argument("--results-out")
    ap.add_argument("--server-log"); ap.add_argument("--bwd-out")
    ap.add_argument("--anchor-wall", type=float, default=None,
                    help="bench timeline_anchor_wall; trims bwd to the timeline phase")
    ap.add_argument("--tl-span", type=float, default=None,
                    help="timeline span (s) for trimming")
    a = ap.parse_args()
    if a.results_in and a.results_out:
        conv_results(a.results_in, a.results_out)
    if a.server_log and a.bwd_out:
        conv_bwd(a.server_log, a.bwd_out, a.anchor_wall, a.tl_span)
