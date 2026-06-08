#!/usr/bin/env python3
"""compare_slora_dserve.py — two-way temporal-sharing comparison figure.

A trimmed-down sibling of ``compare_temporal_both.py``. Instead of three
inference traces it overlays exactly TWO:

  * SLoRA       — inference latency baseline (NO finetune throughput drawn)
  * DeltaServe  — co-serving run (inference latency + FT throughput band)

Layout (2 rows, full width):

  Row 0   Scheduled request timeline (req/s bars + output tok/s line).
  Row 1   Left  y-axis — per-request E2E latency scatter for BOTH runs.
          Right y-axis — DeltaServe finetune (backward) throughput in tok/s,
                         binned from the bwd_log's ``batch_tokens``.

This script is intentionally SELF-CONTAINED — the loaders, the bwd-log
binning, the smoothing and the timeline panel are copied (not imported)
from ``auto_plot.py`` / ``compare_temporal.py`` so the figure can be
produced from a standalone bundle dir with no package on the path. The
math is kept identical to those modules.

Time alignment (matches compare_temporal.py):
  * ``tl_base`` = the timeline file's first ``timestamp_s`` (e.g. 45s). The
    timeline bars and both latency series are placed on this native clock.
  * Each latency CSV is auto-classified as either already-native (min
    ``t_rel_s`` ≈ ``tl_base`` — used verbatim) or normalized-to-0 (min ≈ 0 —
    shifted up by ``tl_base``). Override with ``--slora-offset`` /
    ``--dserve-offset`` if the heuristic guesses wrong.
  * The DeltaServe bwd_log is wall-clock and assumed already trimmed to the
    timeline phase (auto_benchmark's ``trim_bwd_log_before``), so its first
    row anchors at ``tl_base``. Nudge with ``--ft-offset`` if needed.

Usage:
  python compare_slora_dserve.py                       # defaults (this dir)
  python compare_slora_dserve.py --gpu-name "RTX 5090"
  python compare_slora_dserve.py --slora a.csv --dserve b.csv --bwd c.csv
"""
import argparse
import csv
import datetime
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Settings — edit here to retitle / resize / restyle.
# ============================================================================

# ---- Output ----
GENERATE_PDF = True
PNG_DPI = 130

# ---- Figure ----
FIGSIZE = (15, 9)
HEIGHT_RATIOS = (1.0, 1.0)        # timeline row / comparison row
ROW_HSPACE = 0.05
SUPTITLE = None

# ---- Per-panel title ----
PANEL_TITLE_TEMPLATE = "{gpu_name} — E2E latency & FT throughput"

# ---- Axis labels ----
XLABEL = "Time (s)"
YLABEL_LATENCY = "E2E latency (s)"
YLABEL_FT = "FT throughput (tok/s)"

# ---- Y-axis headroom (comparison panel) ----
YMAX_HEADROOM = 1.45

# ---- Display names (legend) ----
DISPLAY_NAME_SLORA = "SLoRA"
DISPLAY_NAME_DSERVE = "DeltaServe-SLoRA"
DISPLAY_NAME_TAIL_OVERHEAD = "Slowest 1% overhead"

# ---- SLoRA latency normalization ----
# SLoRA runs on llama1 (7B) while DeltaServe runs on llama3 (8B). To compare
# the two on the same model footing, scale the SLoRA E2E latency up by this
# factor (llama1→llama3, ~+15% from the larger per-decode weight footprint).
# Only the latency is scaled — request arrival times (t_rel_s) are unchanged.
# Set to 1.0 to disable; override per-run with --slora-latency-factor.
SLORA_LATENCY_FACTOR = 1.15

# ---- Font sizes / weights ----
FONTSIZE_PANEL_TITLE = 20
FONTSIZE_AXIS_LABEL = 14
FONTSIZE_TICK = 10
FONTSIZE_LEGEND = 13
FONTWEIGHT_TITLE = "bold"
FONTWEIGHT_AXIS_LABEL = "bold"

# ---- Legend placement ----
LEGEND_LOC = "upper center"
LEGEND_BBOX_TO_ANCHOR = (0.5, 1.0)
LEGEND_NCOL = 2

# ---- Colors ----
SLORA_COLOR = "tab:blue"
DSERVE_COLOR = "tab:red"
FT_COLOR = "tab:orange"          # matches auto_plot.FT_SHADE_COLOR
FT_FILL_ALPHA = 0.18
FT_LINE_WIDTH = 1.7

# ---- Scatter style ----
SCATTER_SIZE = 10
SCATTER_ALPHA = 0.7

# ============================================================================
# Self-contained helpers (copied verbatim in behaviour from auto_plot.py /
# compare_temporal.py — do NOT import, this script stands alone).
# ============================================================================


def _f(x) -> float:
    try:
        if x is None or str(x).strip() == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def load_results(csv_path: str) -> dict:
    """Columns: idx,t_rel_s,latency_s,status,ttft_s,... -> dict of np arrays
    plus a boolean ``ok`` mask (status == "ok")."""
    cols = {k: [] for k in ["idx", "t_rel_s", "latency_s", "ttft_s"]}
    ok = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        required = ["idx", "t_rel_s", "latency_s", "status", "ttft_s"]
        missing = [c for c in required if c not in (r.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} missing columns: {missing}")
        for row in r:
            cols["idx"].append(_f(row.get("idx")))
            cols["t_rel_s"].append(_f(row.get("t_rel_s")))
            cols["latency_s"].append(_f(row.get("latency_s")))
            cols["ttft_s"].append(_f(row.get("ttft_s")))
            ok.append(str(row.get("status", "")).strip() == "ok")
    out = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    out["ok"] = np.asarray(ok, dtype=bool)
    return out


def load_timeline(csv_path: str) -> dict:
    """Columns: timestamp_s, max_new_tokens (+ extras). Returns t_rel_s
    NORMALIZED to start at 0 (caller adds tl_base back), plus max_new_tokens."""
    ts, tok = [], []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for c in ("timestamp_s", "max_new_tokens"):
            if c not in (r.fieldnames or []):
                raise ValueError(f"{csv_path} missing column: {c}")
        for row in r:
            ts.append(_f(row.get("timestamp_s")))
            tok.append(_f(row.get("max_new_tokens")))
    ts = np.asarray(ts, dtype=float)
    tok = np.asarray(tok, dtype=float)
    t0 = float(ts.min()) if len(ts) else 0.0
    return {"t_rel_s": ts - t0, "max_new_tokens": tok}


def _timeline_base(path: str) -> float:
    """Smallest ``timestamp_s`` in the timeline file — the schedule's first-
    request offset that the benchmark strips when normalizing latency."""
    try:
        with open(path) as f:
            vals = [float(row["timestamp_s"]) for row in csv.DictReader(f)]
    except Exception:
        return 0.0
    return min(vals) if vals else 0.0


def _smooth(y, window_s, bin_s):
    if y.size == 0 or window_s <= 0 or bin_s <= 0:
        return y
    w = max(1, int(round(window_s / bin_s)))
    if w <= 1 or y.size < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def _auto_window(t_max):
    return 5.0 if t_max <= 0 else float(np.clip(t_max / 100.0, 5.0, 60.0))


def plot_request_timeline(ax, tl):
    """req/s (bars, left) + output tok/s (line, right), per integer second.
    Expects ``tl['t_rel_s']`` already on the desired (native) clock."""
    t = tl["t_rel_s"]
    tok = tl["max_new_tokens"]
    if len(t) == 0:
        ax.set_title("Request Timeline (empty)")
        return
    bucket_idx = np.floor(t).astype(int)
    n = int(bucket_idx.max()) + 1
    req_per_s = np.bincount(bucket_idx, minlength=n).astype(float)
    tok_per_s = np.bincount(bucket_idx, weights=tok, minlength=n).astype(float)
    centers = np.arange(n, dtype=float)
    ax.bar(centers, req_per_s, width=1.0, color="tab:gray", alpha=0.55,
           align="edge", label="req/s", edgecolor="none")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Requests / s", color="tab:gray")
    ax.tick_params(axis="y", labelcolor="tab:gray")
    ax.set_xlim(0, n)
    ax.set_ylim(bottom=0)
    ax.set_title("Scheduled Request Timeline")
    ax2 = ax.twinx()
    ax2.plot(centers + 0.5, tok_per_s, color="tab:green", marker="o",
             markersize=3, linewidth=1.5, label="tokens/s")
    ax2.set_ylabel("Output tokens / s", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.set_ylim(bottom=0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)


def _ft_throughput_curve(bwd_path: str, anchor_wall, base: float,
                         bin_s: float = 1.0, window_s=None):
    """Bin each backward's ``batch_tokens`` at its completion time, mapped to
    timeline coords ``base + (completion_wall - anchor_wall)``. Returns
    ``(centers, tok_per_s, t_max)`` smoothed, or ``None`` when no FT data.
    ``anchor_wall=None`` anchors at the first backward row."""
    if not os.path.exists(bwd_path):
        return None
    rows = []
    with open(bwd_path, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "timestamp" not in r.fieldnames:
            return None
        for row in r:
            try:
                dt = datetime.datetime.fromisoformat(
                    (row.get("timestamp") or "").strip())
            except (ValueError, TypeError):
                continue
            rows.append((dt, _f(row.get("batch_tokens"))))
    if not rows:
        return None
    rows.sort(key=lambda x: x[0])
    if anchor_wall is None:
        anchor_wall = rows[0][0]
    coords = np.array([base + (dt - anchor_wall).total_seconds()
                       for dt, _ in rows], dtype=float)
    toks = np.array([bt for _, bt in rows], dtype=float)
    toks[~np.isfinite(toks)] = 0.0
    keep = coords >= 0.0
    coords, toks = coords[keep], toks[keep]
    if coords.size == 0:
        return None
    t_max = float(coords.max())
    n_bins = int(np.ceil((t_max + bin_s) / bin_s))
    per_bin = np.zeros(n_bins, dtype=float)
    idx = np.minimum(np.floor(coords / bin_s).astype(int), n_bins - 1)
    np.add.at(per_bin, idx, toks)
    win_s = window_s if window_s is not None else _auto_window(t_max)
    tok_per_s = _smooth(per_bin / bin_s, win_s, bin_s)
    centers = (np.arange(n_bins) + 0.5) * bin_s
    return centers, tok_per_s, t_max


def _latency_stats(res):
    """Return ``(avg, slow1, n)`` over ok & finite E2E latencies. ``avg`` is
    the full mean; ``slow1`` is the mean of the slowest 1% (>= p99)."""
    ok = res["ok"]
    lat = res["latency_s"][ok]
    lat = lat[np.isfinite(lat)]
    if lat.size == 0:
        return float("nan"), float("nan"), 0
    avg = float(lat.mean())
    p99 = float(np.percentile(lat, 99))
    above99 = lat[lat >= p99]
    slow1 = float(above99.mean()) if above99.size else float(lat.max())
    return avg, slow1, int(lat.size)


def _overhead_pct(a: float, b: float) -> float:
    if np.isfinite(a) and np.isfinite(b) and b > 0:
        return (a - b) / b * 100.0
    return float("nan")


def _dot_handle(color: str) -> Line2D:
    return Line2D([0], [0], marker="o", color="w",
                  markerfacecolor=color, markersize=8)


def _blank_handle() -> Line2D:
    """Invisible spacer handle so a legend row can be padded to full width."""
    return Line2D([0], [0], linestyle="none", marker="")


def _row_major_reorder(items, ncol):
    """Reorder ``items`` so matplotlib's column-major legend fill DISPLAYS
    them in row-major order. Items are assumed to already be laid out
    row-by-row (row 0 first, padded to ``ncol`` per row)."""
    n = len(items)
    if n == 0 or ncol <= 1:
        return list(items)
    nrow = (n + ncol - 1) // ncol
    out = []
    for c in range(ncol):
        for r in range(nrow):
            idx = r * ncol + c
            if idx < n:
                out.append(items[idx])
    return out


def _scatter_latency(ax, res, color: str, x_offset: float = 0.0) -> None:
    ok = res["ok"]
    ax.scatter(res["t_rel_s"][ok] + x_offset, res["latency_s"][ok],
               s=SCATTER_SIZE, color=color, alpha=SCATTER_ALPHA, zorder=3)


def _auto_offset(res_min: float, tl_base: float, name: str) -> float:
    """Classify a results CSV as native (min t_rel_s ≈ tl_base → 0 shift) or
    normalized-to-0 (min ≈ 0 → +tl_base shift), whichever origin it's closer
    to. Prints the decision."""
    if tl_base <= 0:
        return 0.0
    if abs(res_min - tl_base) <= abs(res_min - 0.0):
        print(f"[compare_slora_dserve] {name}: native clock "
              f"(min t_rel_s={res_min:.1f}≈tl_base={tl_base:.1f}) → offset 0")
        return 0.0
    print(f"[compare_slora_dserve] {name}: normalized clock "
          f"(min t_rel_s={res_min:.1f}≈0) → offset +{tl_base:.1f}")
    return tl_base


def _series_min(res) -> float:
    ok = res["ok"]
    t = res["t_rel_s"][ok]
    t = t[np.isfinite(t)]
    return float(t.min()) if t.size else 0.0


# ============================================================================
# Panel + figure
# ============================================================================


def plot_panel(ax, slora, dserve, bwd_path, tl_base, gpu_name,
               slora_offset=None, dserve_offset=None,
               ft_offset=0.0, window_s=None, slora_latency_factor=1.0) -> float:
    """Two-series latency overlay (left) + DeltaServe FT throughput (right).
    Returns the right-most x so the caller can pin a shared x-limit."""
    if slora_offset is None:
        slora_offset = _auto_offset(_series_min(slora), tl_base,
                                    DISPLAY_NAME_SLORA) if slora else 0.0
    if dserve_offset is None:
        dserve_offset = _auto_offset(_series_min(dserve), tl_base,
                                     DISPLAY_NAME_DSERVE) if dserve else 0.0

    t_ends = []
    slora_avg = slora_slow1 = float("nan")
    dserve_avg = dserve_slow1 = float("nan")

    if slora is not None:
        slora_avg, slora_slow1, _ = _latency_stats(slora)
        _scatter_latency(ax, slora, SLORA_COLOR, x_offset=slora_offset)
        m = slora["ok"]
        t_ends.append(slora["t_rel_s"][m] + slora["latency_s"][m] + slora_offset)
    if dserve is not None:
        dserve_avg, dserve_slow1, _ = _latency_stats(dserve)
        _scatter_latency(ax, dserve, DSERVE_COLOR, x_offset=dserve_offset)
        m = dserve["ok"]
        t_ends.append(dserve["t_rel_s"][m] + dserve["latency_s"][m] + dserve_offset)

    ax.set_xlabel(XLABEL, fontsize=FONTSIZE_AXIS_LABEL,
                  fontweight=FONTWEIGHT_AXIS_LABEL)
    ax.set_ylabel(YLABEL_LATENCY, fontsize=FONTSIZE_AXIS_LABEL,
                  fontweight=FONTWEIGHT_AXIS_LABEL)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    ax.set_ylim(bottom=0)
    if PANEL_TITLE_TEMPLATE:
        ax.set_title(PANEL_TITLE_TEMPLATE.format(gpu_name=gpu_name),
                     fontsize=FONTSIZE_PANEL_TITLE, fontweight=FONTWEIGHT_TITLE)
    ax.grid(True, alpha=0.25)

    t_max = max((float(np.nanmax(a)) for a in t_ends if len(a)), default=0.0)

    # ---- right axis: DeltaServe FT throughput (anchored at tl_base) ----
    ax_r = ax.twinx()
    ax_r.set_ylabel(YLABEL_FT, color=FT_COLOR, fontsize=FONTSIZE_AXIS_LABEL,
                    fontweight=FONTWEIGHT_AXIS_LABEL)
    ax_r.tick_params(axis="y", labelcolor=FT_COLOR, labelsize=FONTSIZE_TICK)

    ft_peak = 0.0
    ft_mean = 0.0
    base = (dserve_offset if dserve is not None else tl_base) + ft_offset
    ft = _ft_throughput_curve(bwd_path, None, base, window_s=window_s)
    if ft is None:
        ax_r.set_yticks([])
    else:
        centers, tok_per_s, ft_last = ft
        ax_r.fill_between(centers, 0, tok_per_s, color=FT_COLOR,
                          alpha=FT_FILL_ALPHA, linewidth=0, zorder=1)
        ax_r.plot(centers, tok_per_s, color=FT_COLOR, linewidth=FT_LINE_WIDTH,
                  zorder=2)
        ft_peak = float(np.nanmax(tok_per_s)) if tok_per_s.size else 0.0
        # Mean over the WHOLE curve (all bins from 0, incl. leading zeros
        # before FT starts) — matches compare_temporal_both._draw_ft_curve.
        ft_mean = float(np.mean(tok_per_s)) if tok_per_s.size else 0.0
        t_max = max(t_max, ft_last)
        print(f"[compare_slora_dserve] {gpu_name}: FT peak={ft_peak:.0f} "
              f"tok/s, mean={ft_mean:.0f} tok/s, t_last={ft_last:.1f}s")

    # ---- headroom on both axes ----
    lat_peaks = []
    for r in (slora, dserve):
        if r is None:
            continue
        m = r["ok"]
        if m.any():
            lat_peaks.append(float(np.nanmax(r["latency_s"][m])))
    lat_peak = max((p for p in lat_peaks if np.isfinite(p) and p > 0), default=0.0)
    if lat_peak > 0:
        ax.set_ylim(0, lat_peak * YMAX_HEADROOM)
    if ft_peak > 0:
        ax_r.set_ylim(0, ft_peak * YMAX_HEADROOM)
    else:
        ax_r.set_ylim(bottom=0)

    # ---- legend: row 1 = latency entries, row 2 = FT throughput ----
    lat_h, lat_l = [], []
    if slora is not None and np.isfinite(slora_avg):
        lat_h.append(_dot_handle(SLORA_COLOR))
        scale_note = (f", ×{slora_latency_factor:g}"
                      if abs(slora_latency_factor - 1.0) > 1e-9 else "")
        lat_l.append(f"{DISPLAY_NAME_SLORA} (avg {slora_avg:.3f}s{scale_note})")
    if dserve is not None and np.isfinite(dserve_avg):
        parts = [f"avg {dserve_avg:.3f}s"]
        avg_oh = _overhead_pct(dserve_avg, slora_avg)
        if np.isfinite(avg_oh):
            parts.append(f"{avg_oh:+.1f}%")
        lat_h.append(_dot_handle(DSERVE_COLOR))
        lat_l.append(f"{DISPLAY_NAME_DSERVE} ({', '.join(parts)})")
    ft_h, ft_l = [], []
    if ft_mean > 0:
        ft_h.append(Patch(facecolor=FT_COLOR, alpha=FT_FILL_ALPHA,
                          edgecolor=FT_COLOR))
        ft_l.append(f"{DISPLAY_NAME_DSERVE} FT Throughput: {ft_mean:.0f} tok/s")

    # Stack the two groups row-by-row (each padded to LEGEND_NCOL with blank
    # spacers) so latency sits on row 1 and throughput on row 2, then permute
    # for matplotlib's column-major fill.
    rows_h, rows_l = [], []
    for grp_h, grp_l in ((lat_h, lat_l), (ft_h, ft_l)):
        if not grp_h:
            continue
        rows_h.extend(grp_h)
        rows_l.extend(grp_l)
        while len(rows_h) % LEGEND_NCOL != 0:
            rows_h.append(_blank_handle())
            rows_l.append("")
    if rows_h:
        legend_kwargs = dict(loc=LEGEND_LOC, fontsize=FONTSIZE_LEGEND,
                             ncol=LEGEND_NCOL)
        if LEGEND_BBOX_TO_ANCHOR is not None:
            legend_kwargs["bbox_to_anchor"] = LEGEND_BBOX_TO_ANCHOR
        ax.legend(_row_major_reorder(rows_h, LEGEND_NCOL),
                  _row_major_reorder(rows_l, LEGEND_NCOL), **legend_kwargs)
    return t_max


def build_figure(timeline_path, slora_path, dserve_path, bwd_path, gpu_name,
                 slora_offset=None, dserve_offset=None,
                 ft_offset=0.0, window_s=None,
                 slora_latency_factor=1.0) -> plt.Figure:
    row_unit = FIGSIZE[1] / sum(HEIGHT_RATIOS)
    fig = plt.figure(figsize=(FIGSIZE[0], row_unit * sum(HEIGHT_RATIOS)),
                     constrained_layout=True)
    gs_kwargs = dict(figure=fig, height_ratios=list(HEIGHT_RATIOS))
    if ROW_HSPACE is not None:
        gs_kwargs["hspace"] = ROW_HSPACE
    gs = GridSpec(2, 1, **gs_kwargs)

    # Row 0: scheduled request timeline (native clock).
    ax_tl = fig.add_subplot(gs[0, 0])
    tl_base = _timeline_base(timeline_path) if os.path.exists(timeline_path) else 0.0
    tl_max = 0.0
    if os.path.exists(timeline_path):
        tl = load_timeline(timeline_path)
        tl["t_rel_s"] = tl["t_rel_s"] + tl_base   # undo normalize → native
        tl_max = float(tl["t_rel_s"].max()) if len(tl["t_rel_s"]) else 0.0
        plot_request_timeline(ax_tl, tl)
        if ax_tl.get_title():
            ax_tl.title.set_fontsize(FONTSIZE_PANEL_TITLE)
            ax_tl.title.set_fontweight(FONTWEIGHT_TITLE)
        for lbl in (ax_tl.xaxis.label, ax_tl.yaxis.label):
            lbl.set_fontsize(FONTSIZE_AXIS_LABEL)
            lbl.set_fontweight(FONTWEIGHT_AXIS_LABEL)
        for child in ax_tl.figure.axes:
            if child is ax_tl:
                continue
            if (child.get_position().bounds == ax_tl.get_position().bounds
                    and child.yaxis.label.get_text()):
                child.yaxis.label.set_fontsize(FONTSIZE_AXIS_LABEL)
                child.yaxis.label.set_fontweight(FONTWEIGHT_AXIS_LABEL)
    else:
        ax_tl.text(0.5, 0.5, f"{os.path.basename(timeline_path)}\n(not found)",
                   ha="center", va="center", transform=ax_tl.transAxes,
                   fontsize=12, color="0.55")
        ax_tl.set_xticks([])
        ax_tl.set_yticks([])
        ax_tl.set_title("Scheduled Request Timeline (missing)",
                        fontsize=FONTSIZE_PANEL_TITLE, fontweight=FONTWEIGHT_TITLE)

    # Row 1: comparison panel.
    slora = load_results(slora_path) if os.path.exists(slora_path) else None
    dserve = load_results(dserve_path) if os.path.exists(dserve_path) else None
    if slora is None:
        print(f"[compare_slora_dserve] WARN: SLoRA file not found: {slora_path}",
              file=sys.stderr)
    if dserve is None:
        print(f"[compare_slora_dserve] WARN: DeltaServe file not found: {dserve_path}",
              file=sys.stderr)

    # Normalize SLoRA (llama1) E2E latency onto the DeltaServe (llama3) model
    # footing. Only latency is scaled; arrival times (t_rel_s) are the shared
    # schedule and stay put.
    if slora is not None and abs(slora_latency_factor - 1.0) > 1e-9:
        slora["latency_s"] = slora["latency_s"] * slora_latency_factor
        print(f"[compare_slora_dserve] SLoRA latency scaled ×{slora_latency_factor:g} "
              f"(llama1→llama3 normalization)")

    ax = fig.add_subplot(gs[1, 0])
    panel_tmax = plot_panel(
        ax, slora, dserve, bwd_path, tl_base, gpu_name,
        slora_offset=slora_offset, dserve_offset=dserve_offset,
        ft_offset=ft_offset, window_s=window_s,
        slora_latency_factor=slora_latency_factor)

    t_max = max(panel_tmax, tl_max)
    if t_max > 0:
        xlim = (0, t_max * 1.01)
        ax_tl.set_xlim(*xlim)
        if ax.has_data():
            ax.set_xlim(*xlim)

    if SUPTITLE:
        fig.suptitle(SUPTITLE, fontsize=14)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeline", default=os.path.join(_HERE, "timeline_nutanix.csv"),
                    help="Scheduled-request timeline CSV (timestamp_s, "
                         "max_new_tokens). Default: ./timeline_nutanix.csv.")
    ap.add_argument("--slora", default=os.path.join(_HERE, "latency_slora.csv"),
                    help="SLoRA results CSV (inference latency baseline). "
                         "Default: ./latency_slora.csv.")
    ap.add_argument("--dserve",
                    default=os.path.join(_HERE,
                                         "timeline_results_decode_prefill_bwd_nutanix.csv"),
                    help="DeltaServe results CSV (co-serving inference "
                         "latency). Default: ./timeline_results_decode_prefill_bwd_nutanix.csv.")
    ap.add_argument("--bwd",
                    default=os.path.join(_HERE, "bwd_log_decode_prefill_bwd_nutanix.csv"),
                    help="DeltaServe backward log CSV (FT throughput source). "
                         "Default: ./bwd_log_decode_prefill_bwd_nutanix.csv.")
    ap.add_argument("--gpu-name", default="RTX 5090",
                    help="GPU display name used in the panel title.")
    ap.add_argument("--slora-offset", type=float, default=None,
                    help="Manual x-shift (s) for the SLoRA latency series. "
                         "Default: auto (native vs normalized detection).")
    ap.add_argument("--dserve-offset", type=float, default=None,
                    help="Manual x-shift (s) for the DeltaServe latency "
                         "series. Default: auto.")
    ap.add_argument("--ft-offset", type=float, default=0.0,
                    help="Extra manual shift (s) of the FT-throughput curve. "
                         "Default 0.")
    ap.add_argument("--window", type=float, default=None,
                    help="FT-throughput smoothing window (s). Default: auto "
                         "(~t_max/100, clamped 5–60s).")
    ap.add_argument("--slora-latency-factor", type=float, default=SLORA_LATENCY_FACTOR,
                    help="Multiply SLoRA E2E latency by this factor to put it on "
                         "the DeltaServe (llama3) model footing — SLoRA runs "
                         f"llama1. Default: {SLORA_LATENCY_FACTOR}. Use 1.0 to disable.")
    ap.add_argument("--output", default=None,
                    help="Output PNG path. Default: "
                         "<dir>/compare_slora_dserve.png next to --dserve.")
    args = ap.parse_args()

    out_path = args.output or os.path.join(_HERE, "compare_slora_dserve.png")

    fig = build_figure(
        args.timeline, args.slora, args.dserve, args.bwd, args.gpu_name,
        slora_offset=args.slora_offset, dserve_offset=args.dserve_offset,
        ft_offset=args.ft_offset, window_s=args.window,
        slora_latency_factor=args.slora_latency_factor)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=PNG_DPI)
    print(f"[compare_slora_dserve] wrote figure → {out_path}")
    if GENERATE_PDF:
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0)
        print(f"[compare_slora_dserve] wrote figure → {pdf_path}  (no-margin)")
    plt.close(fig)


if __name__ == "__main__":
    main()
