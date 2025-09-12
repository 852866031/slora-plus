#!/usr/bin/env python3
"""
plot_sequence_boxes.py
Render sequence (Gantt-style) plots from an Nsight Systems SQLite export.

Each event is a rectangle whose length equals its duration. We create two plots:
  1) First decode that overlaps job_66:backward
  2) Control decode (trimmed to start just after backward ends)

Default lanes (plotted top→bottom):
  - s7_kern    : CUPTI kernels on streamId=7   (decode)
  - s7_sync    : Synchronization on streamId=7
  - s7_memcpy  : Memcpys on streamId=7
  - rt_s7      : CUDA runtime API attributed to streamId=7 (if available) else all
  - s13_kern   : CUPTI kernels on streamId=13  (backward)
  - s13_memcpy : Memcpys on streamId=13

You can pick which lanes to draw via --lanes; order there controls vertical order.

Usage:
  python plot_sequence_boxes.py your_trace_cleaned.sqlite --out seq_plots
  # Optional:
  python plot_sequence_boxes.py your_trace_cleaned.sqlite --lanes s7_kern,s7_sync,rt_s7 --limit_per_lane 10000 --stride 2
"""

import argparse
import math
import os
import sqlite3
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

CAND_START = ["start", "start_ns", "startTime", "startTimestamp"]
CAND_END   = ["end", "end_ns", "endTime", "endTimestamp", '"end"']

DEFAULT_LANES = ["s7_kern","s7_sync","s7_memcpy","rt_s7","s13_kern","s13_memcpy"]

def table_exists(cur, name, schema="main"):
    cur.execute(f"SELECT COUNT(*) FROM {schema}.sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone()[0] == 1

def colnames(cur, table):
    cur.execute(f"PRAGMA table_info('{table}')")
    return [r[1] for r in cur.fetchall()]

def find_time_cols(cur, table) -> Tuple[Optional[str], Optional[str]]:
    cols = colnames(cur, table)
    s = next((c for c in CAND_START if c in cols), None)
    e = next((c for c in CAND_END   if c in cols), None)
    return s, e

def has_col(cur, table, col) -> bool:
    return col in colnames(cur, table)

def find_backward_window(cur):
    cur.execute("""
        SELECT e.start AS s, e."end" AS e
        FROM NVTX_EVENTS e LEFT JOIN StringIds sid ON e.textId = sid.id
        WHERE COALESCE(e.text, sid.value) GLOB 'job_66:backward*'
        ORDER BY (e."end" - e.start) DESC
        LIMIT 1
    """)
    row = cur.fetchone()
    if not row:
        raise RuntimeError("Could not find NVTX range for 'job_66:backward'")
    return int(row["s"]), int(row["e"])

def find_decode_windows(cur):
    cur.execute("""
        SELECT e.start AS s, e."end" AS e, COALESCE(e.text, sid.value) AS label
        FROM NVTX_EVENTS e LEFT JOIN StringIds sid ON e.textId = sid.id
        WHERE COALESCE(e.text, sid.value) GLOB 'job_6[7-9]:decode*'
           OR COALESCE(e.text, sid.value) GLOB 'job_7[0-1]:decode*'
        ORDER BY s
    """)
    return [(int(r["s"]), int(r["e"]), r["label"]) for r in cur.fetchall()]

def pick_overlapped_and_control(backward, decodes):
    bS, bE = backward
    overlapped, nonover = [], []
    for (s, e, lbl) in decodes:
        if min(e, bE) - max(s, bS) > 0:
            overlapped.append((s, e, lbl))
        else:
            nonover.append((s, e, lbl))
    if not overlapped:
        raise RuntimeError("No decode window overlaps job_66:backward.")
    overlapped.sort(key=lambda x: x[0])
    first_overlap = overlapped[0]

    # Prefer non-overlap after backward end
    control = None
    after_nonover = [d for d in nonover if d[0] >= bE]
    preferred = [d for d in after_nonover if "job_71:decode" in d[2]]
    candidates = preferred if preferred else after_nonover
    if candidates:
        control = min(candidates, key=lambda d: abs(d[0] - bE))

    if control is None:
        # Trim the first decode that crosses bE
        crossing = [(s, e, lbl) for (s, e, lbl) in decodes if e > bE]
        if not crossing:
            raise RuntimeError("Could not construct a control decode: no decode extends past backward end.")
        crossing.sort(key=lambda d: d[0])
        s, e, lbl = crossing[0]
        safe_start = max(s, bE + 100_000)  # +0.1 ms
        if safe_start >= e:
            # try next
            for (s2, e2, lbl2) in crossing[1:]:
                safe2 = max(s2, bE + 100_000)
                if safe2 < e2:
                    control = (safe2, e2, lbl2 + " [trimmed]"); break
        else:
            control = (safe_start, e, lbl + " [trimmed]")

    return first_overlap, control

def fetch_events(cur, table, window, stream_id=None, select_bytes=False):
    s_col, e_col = find_time_cols(cur, table)
    if not s_col or not e_col or not table_exists(cur, table):
        return []
    where = f"({s_col} < ? AND {e_col} > ?)"
    params = [window[1], window[0]]
    if stream_id is not None and has_col(cur, table, "streamId"):
        where += " AND streamId = ?"
        params.append(stream_id)
    if select_bytes and has_col(cur, table, "bytes"):
        cur.execute(f"SELECT {s_col} AS s, {e_col} AS e, bytes FROM {table} WHERE {where}", params)
        return [(int(r["s"]), int(r["e"]), int(r["bytes"])) for r in cur.fetchall()]
    else:
        cur.execute(f"SELECT {s_col} AS s, {e_col} AS e FROM {table} WHERE {where}", params)
        return [(int(r["s"]), int(r["e"])) for r in cur.fetchall()]

def build_lane_events(con, window, lanes: List[str], max_events_per_lane: int, stride: int):
    cur = con.cursor()
    cur.row_factory = sqlite3.Row

    lane_to_events = {}

    # Define how each lane maps to a table & filter
    # (We only need start/end to draw rectangles.)
    for lane in lanes:
        if lane == "s7_kern":
            ev = fetch_events(cur, "CUPTI_ACTIVITY_KIND_KERNEL", window, stream_id=7)
        elif lane == "s13_kern":
            ev = fetch_events(cur, "CUPTI_ACTIVITY_KIND_KERNEL", window, stream_id=13)
        elif lane == "s7_memcpy":
            ev = fetch_events(cur, "CUPTI_ACTIVITY_KIND_MEMCPY", window, stream_id=7)
        elif lane == "s13_memcpy":
            ev = fetch_events(cur, "CUPTI_ACTIVITY_KIND_MEMCPY", window, stream_id=13)
        elif lane == "s7_sync":
            ev = fetch_events(cur, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", window, stream_id=7)
        elif lane == "s13_sync":
            ev = fetch_events(cur, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", window, stream_id=13)
        elif lane == "rt_s7":
            # If runtime has streamId, filter to 7; otherwise take all runtime API in the window
            table = "CUPTI_ACTIVITY_KIND_RUNTIME"
            has_rt = table_exists(cur, table)
            if not has_rt:
                ev = []
            else:
                if has_col(cur, table, "streamId"):
                    ev = fetch_events(cur, table, window, stream_id=7)
                else:
                    ev = fetch_events(cur, table, window, stream_id=None)
        elif lane == "rt_all":
            table = "CUPTI_ACTIVITY_KIND_RUNTIME"
            ev = fetch_events(cur, table, window, stream_id=None) if table_exists(cur, table) else []
        else:
            # Unknown lane name -> empty
            ev = []

        # Normalize to relative milliseconds; optionally stride/subsample to keep PNG size reasonable
        rel = [((s - window[0]) / 1e6, (e - window[0]) / 1e6) for (s, e, *_) in ev if e > s]
        rel.sort(key=lambda t: t[0])
        if stride > 1:
            rel = rel[::stride]
        if max_events_per_lane > 0 and len(rel) > max_events_per_lane:
            rel = rel[:max_events_per_lane]

        lane_to_events[lane] = rel

    return lane_to_events

def plot_sequence(window, lanes: List[str], lane_to_events, title, out_png):
    total_ms = (window[1] - window[0]) / 1e6
    fig, ax = plt.subplots(figsize=(12, 1.2 + 0.45 * max(1, len(lanes))))  # compact height

    y_ticks = []
    y_labels = []
    for idx, lane in enumerate(lanes):
        y_center = len(lanes) - 1 - idx  # top-to-bottom order
        y_ticks.append(y_center)
        y_labels.append(lane)
        events = lane_to_events.get(lane, [])
        # draw rectangles
        for (start_ms, end_ms) in events:
            width = max(end_ms - start_ms, 0.0005)  # tiny minimum width so zero-lengths still visible
            rect = Rectangle((start_ms, y_center - 0.35), width, 0.7, linewidth=0.5, fill=True, alpha=0.7)
            ax.add_patch(rect)

    ax.set_ylim(-1, len(lanes))
    ax.set_xlim(0, total_ms)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time from decode start (ms)")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Sequence (Gantt) plot for overlapped vs control decode windows.")
    ap.add_argument("sqlite_path", help="Nsight Systems .sqlite")
    ap.add_argument("--out", default="seq_plots", help="Output directory for PNGs")
    ap.add_argument("--lanes", default=",".join(DEFAULT_LANES),
                    help=f"Comma-separated lanes to plot (default: {','.join(DEFAULT_LANES)})\n"
                         "Choices include: s7_kern,s13_kern,s7_memcpy,s13_memcpy,s7_sync,s13_sync,rt_s7,rt_all")
    ap.add_argument("--limit_per_lane", type=int, default=0,
                    help="Max rectangles per lane (0 = no limit). Useful to keep files small.")
    ap.add_argument("--stride", type=int, default=1,
                    help="Take every Nth event per lane (default 1 = keep all).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]

    con = sqlite3.connect(args.sqlite_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # basic checks
    for t in ("NVTX_EVENTS","StringIds"):
        if not table_exists(cur, t):
            raise RuntimeError(f"Required table missing: {t}")

    bS, bE = find_backward_window(cur)
    decodes = find_decode_windows(cur)
    (oS, oE, oLbl), (cS, cE, cLbl) = pick_overlapped_and_control((bS,bE), decodes)

    print(f"Backward window: {bS}..{bE} ({(bE-bS)/1e6:.3f} ms)")
    print(f"Overlapped decode: {oLbl} {oS}..{oE} ({(oE-oS)/1e6:.3f} ms)")
    print(f"Control decode:    {cLbl} {cS}..{cE} ({(cE-cS)/1e6:.3f} ms)")

    # Build & plot overlapped
    lane_events_overlap = build_lane_events(con, (oS, oE), lanes, args.limit_per_lane, args.stride)
    plot_sequence((oS, oE), lanes, lane_events_overlap,
                  f"Overlapped decode ({oLbl}) vs backward", os.path.join(args.out, "sequence_overlapped.png"))

    # Build & plot control
    lane_events_control = build_lane_events(con, (cS, cE), lanes, args.limit_per_lane, args.stride)
    plot_sequence((cS, cE), lanes, lane_events_control,
                  f"Control decode ({cLbl})", os.path.join(args.out, "sequence_control.png"))

    con.close()
    print(f"Saved: {os.path.join(args.out, 'sequence_overlapped.png')} and sequence_control.png")

if __name__ == "__main__":
    main()