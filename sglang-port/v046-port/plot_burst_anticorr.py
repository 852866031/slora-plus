#!/usr/bin/env python3
"""Plot the FT-throughput-vs-inference-load anti-correlation on a bursty timeline.

Substitute for the plan's proprietary nutanix success-criterion plot. Top panel:
scheduled inference RPS over time (the load). Bottom panel: FT throughput
(tokens/s, reconstructed from the server log's `real_backward ... cum_tokens=`
deltas) + per-request E2E latency. With the RPS throttle (or SLO gate) ON, FT
throughput should FILL the quiet windows and BACK OFF during the burst.

Usage:
  python plot_burst_anticorr.py --server-log output/server_tight_co.log \
      --results output/timeline_results_tight_co_real_sub.csv \
      --timeline ../../eval/llama3/timelines/BURST/timeline_tight.csv \
      --out plots/burst_anticorr_on.png --title "RPS throttle ON"
"""
import argparse, csv, re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TS = re.compile(r"real_backward #\d+:.*cum_tokens=(\d+)\s+wall=([\d.]+)")


def parse_anchor_wall(bench_log):
    """Wall-clock (time.time) of the timeline anchor, logged by the bench."""
    for line in open(bench_log, errors="ignore"):
        m = re.search(r"timeline_anchor_wall=([\d.]+)", line)
        if m:
            return float(m.group(1))
    return None


def parse_ft(server_log):
    """[(fire_wall_epoch, cum_tokens)] from the `cum_tokens=N wall=<epoch>` lines."""
    out = []
    for line in open(server_log, errors="ignore"):
        m = _TS.search(line)
        if m:
            out.append((float(m.group(2)), int(m.group(1))))
    return out


def ft_throughput_per_s(fires, anchor_wall):
    """tokens/s per integer TIMELINE second (fire_wall - anchor_wall). Falls back
    to zeroing at the first fire if no anchor (less precise)."""
    if not fires:
        return {}
    base = anchor_wall if anchor_wall is not None else fires[0][0]
    prev_tok = 0
    by_s = defaultdict(float)
    for t, cum in fires:
        delta = max(0, cum - prev_tok); prev_tok = cum
        by_s[int(t - base)] += delta
    return by_s


def inference_rps(timeline_csv):
    """req/s per integer TIMELINE second (absolute timeline clock, not zeroed)."""
    by_s = defaultdict(int)
    for r in csv.DictReader(open(timeline_csv)):
        by_s[int(float(r["timestamp_s"]))] += 1
    return by_s


def latency_points(results_csv):
    """(sent_t on timeline clock, latency ms) for inference reqs."""
    xs, ys = [], []
    if not Path(results_csv).exists():
        return xs, ys
    for r in csv.DictReader(open(results_csv)):
        if r.get("is_ft") == "1" or not r.get("latency_s") or r.get("error"):
            continue
        xs.append(float(r["sent_t"]))
        ys.append(float(r["latency_s"]) * 1000.0)
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--bench-log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    anchor = parse_anchor_wall(a.bench_log)
    rps = inference_rps(a.timeline)
    fires = parse_ft(a.server_log)
    ftps = ft_throughput_per_s(fires, anchor)
    lx, ly = latency_points(a.results)

    tmax = max(max(rps, default=0), max(ftps, default=0), int(max(lx, default=0))) + 1
    secs = list(range(tmax))
    rps_y = [rps.get(s, 0) for s in secs]
    ft_y = [ftps.get(s, 0.0) for s in secs]
    total_ft = fires[-1][1] if fires else 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.bar(secs, rps_y, width=0.9, color="#888", label="inference req/s (scheduled)")
    ax1.set_ylabel("inference req/s"); ax1.legend(loc="upper right")
    ax1.set_title(f"Bursty co-serving — {a.title}   (total FT trained: {total_ft} tok)")

    ax2.fill_between(secs, ft_y, step="mid", alpha=0.45, color="#2ca02c",
                     label="FT throughput (tok/s, from backward cum_tokens)")
    ax2.set_ylabel("FT tokens/s", color="#2ca02c"); ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_xlabel("timeline seconds (FT fires aligned to inference clock via anchor_wall)")
    axr = ax2.twinx()
    axr.scatter(lx, ly, s=10, color="#d62728", alpha=0.6, label="E2E latency (ms)")
    axr.set_ylabel("E2E latency (ms)", color="#d62728"); axr.tick_params(axis="y", labelcolor="#d62728")
    l2, lab2 = ax2.get_legend_handles_labels(); lr, labr = axr.get_legend_handles_labels()
    ax2.legend(l2 + lr, lab2 + labr, loc="upper right")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=120)
    print(f"wrote {a.out}  (FT total {total_ft} tok, {len(fires)} fires)")


if __name__ == "__main__":
    main()
