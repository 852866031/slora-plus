#!/usr/bin/env python3
"""
Compare TTFT / Avg TBT between two latency CSV files on a single percentile plot,
with dashed horizontal SLO reference lines loaded from a JSON.

Legend: ONLY curves (TTFT/Avg TBT for both runs).
SLOs: annotated directly on their horizontal lines (not in legend).

CSV columns expected:
  idx,t_rel_s,latency_s,status,ttft_s,avg_tbt_s,worst_tbt_s

SLO JSON example:
{
  "ttft_slo": 0.3,
  "avg_tbt_slo": 0.15,
  "max_tbt_slo": 0.5
}
"""

import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------ Utilities ------------------ #
def _percentile_curve(series: pd.Series) -> np.ndarray:
    """Return y-values for percentiles 0..100 from a float series."""
    vals = series.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([])
    pct = np.linspace(0, 100, 101)
    return np.percentile(vals, pct)


def _safe_series(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name].dropna() if name in df.columns else pd.Series([], dtype=float)


def load_slos(json_path: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Read SLOs from JSON file.
    Keys supported:
      - ttft_slo
      - avg_tbt_slo
      - max_tbt_slo (ignored for this plot)
    """
    if not json_path:
        return None, None
    try:
        with open(json_path, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to read SLO json '{json_path}': {e}")
        return None, None

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    ttft = _to_float(cfg.get("ttft_slo"))
    avg = _to_float(cfg.get("avg_tbt_slo"))
    return ttft, avg


# ------------------ Plot Function ------------------ #
def plot_two_tbt_files(
    file1: str,
    label1: str,
    file2: str,
    label2: str,
    slo_json: Optional[str] = None,
    out_png: str = "compare_ttft_tbt_slo.png",
):
    # Load SLOs (seconds)
    ttft_slo, avg_tbt_slo = load_slos(slo_json)
    if slo_json:
        print(f"Loaded SLOs from {slo_json}: ttft={ttft_slo}, avg_tbt={avg_tbt_slo}")

    # Load and filter
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df1_ok = df1[df1["status"] == "ok"].copy() if "status" in df1.columns else df1.copy()
    df2_ok = df2[df2["status"] == "ok"].copy() if "status" in df2.columns else df2.copy()

    pct = np.linspace(0, 100, 101)
    fig, ax = plt.subplots(figsize=(10, 6))

    # TTFT curves
    y1_ttft = _percentile_curve(_safe_series(df1_ok, "ttft_s"))
    y2_ttft = _percentile_curve(_safe_series(df2_ok, "ttft_s"))
    if y1_ttft.size:
        ax.plot(pct, y1_ttft, color="tab:blue", linestyle="-", label=f"TTFT ({label1})")
    if y2_ttft.size:
        ax.plot(pct, y2_ttft, color="tab:blue", linestyle="--", label=f"TTFT ({label2})")

    # Avg TBT curves
    y1_avg = _percentile_curve(_safe_series(df1_ok, "avg_tbt_s"))
    y2_avg = _percentile_curve(_safe_series(df2_ok, "avg_tbt_s"))
    if y1_avg.size:
        ax.plot(pct, y1_avg, color="tab:orange", linestyle="-", label=f"Avg TBT ({label1})")
    if y2_avg.size:
        ax.plot(pct, y2_avg, color="tab:orange", linestyle="--", label=f"Avg TBT ({label2})")

    # Horizontal SLO lines (with inline annotations on the lines)
    # Place annotation text at the far right of the axes (x ~ 0.99), anchored to the SLO y in data coords.
    if ttft_slo is not None:
        ax.axhline(ttft_slo, color="tab:blue", linestyle=":", linewidth=1.2, alpha=0.9)
        ax.annotate(
            f"TTFT SLO = {ttft_slo:.3f}s",
            xy=(0.995, ttft_slo), xycoords=("axes fraction", "data"),
            xytext=(-2, 0), textcoords="offset points",
            ha="right", va="center",
            fontsize=10, color="tab:blue",
            bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.75),
        )
    if avg_tbt_slo is not None:
        ax.axhline(avg_tbt_slo, color="tab:orange", linestyle=":", linewidth=1.2, alpha=0.9)
        ax.annotate(
            f"Avg TBT SLO = {avg_tbt_slo:.3f}s",
            xy=(0.995, avg_tbt_slo), xycoords=("axes fraction", "data"),
            xytext=(-2, 0), textcoords="offset points",
            ha="right", va="center",
            fontsize=10, color="tab:orange",
            bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.75),
        )

    # Percentile guides
    for p in (50, 90, 99):
        ax.axvline(p, linestyle=":", color="gray", linewidth=1, alpha=0.4)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Seconds")
    ax.set_title(f"TTFT / Avg TBT — {label1} vs {label2}")
    ax.grid(True, alpha=0.3)

    # Legend: curves only
    ax.legend(loc="upper left", ncol=2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"✅ Saved TTFT/TBT comparison plot to {out_png}")


# ------------------ Example Usage ------------------ #
if __name__ == "__main__":
    plot_two_tbt_files(
        file1="latency_co-serving.csv",
        label1="Co-serving",
        file2="latency_inference.csv",
        label2="Inference",
        slo_json="/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/eval/config/finetuning_config.json",
        out_png="compare_ttft_tbt_slo.png",
    )