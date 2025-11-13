import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------
# Utility Functions
# ------------------------------------------------------
def _safe_series(df, name):
    return df[name].dropna() if name in df.columns else pd.Series([], dtype=float)

def _percentile_curve(series):
    vals = series.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([])
    pct = np.linspace(0, 100, 101)
    return np.percentile(vals, pct)

def _safe_label(label: str) -> str:
    return label.replace("_", " ")

# ------------------------------------------------------
# Figure 1–2: Latency Percentile Comparisons
# ------------------------------------------------------
def plot_latency_panels(ax_ttft, ax_latency, files, labels, colors):
    dfs = [pd.read_csv(f) for f in files]
    dfs = [df[df.get("status", "ok") == "ok"] for df in dfs]
    pct = np.linspace(0, 100, 101)
    all_vals = []

    # --- Plot TTFT / Avg TBT ---
    for df, label, color in zip(dfs, labels, colors):
        y_ttft = _percentile_curve(_safe_series(df, "ttft_s"))
        y_avg = _percentile_curve(_safe_series(df, "avg_tbt_s"))
        if y_ttft.size:
            ax_ttft.plot(pct, y_ttft, color=color, linestyle="-", label=f"TTFT ({_safe_label(label)})")
            all_vals.append(y_ttft)
        if y_avg.size:
            ax_ttft.plot(pct, y_avg, color=color, linestyle="--", label=f"Avg TBT ({_safe_label(label)})")
            all_vals.append(y_avg)

    # --- Dynamic Y limit ---
    all_vals = np.concatenate(all_vals) if all_vals else np.array([])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size:
        ymax_cap = np.percentile(all_vals, 80)
        hard_max = all_vals.max()
        ymax_cap = (hard_max + ymax_cap) / 2
        ax_ttft.set_ylim(top=ymax_cap)
        if hard_max > ymax_cap:
            ax_ttft.annotate(
                f"up to {hard_max:.3f}s",
                xy=(0.98, ymax_cap),
                xycoords=("axes fraction", "data"),
                ha="right", va="bottom",
                fontsize=9,
                color="gray",
                bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7),
            )

    ax_ttft.set_title("TTFT / Avg TBT", fontsize=11, fontweight="bold")
    ax_ttft.set_xlabel("Percentile")
    ax_ttft.set_ylabel("Seconds")
    ax_ttft.grid(True, alpha=0.3)
    ax_ttft.legend(fontsize=8)

    # --- Plot Total Latency ---
    for df, label, color in zip(dfs, labels, colors):
        y_lat = _percentile_curve(_safe_series(df, "latency_s"))
        if y_lat.size:
            ax_latency.plot(pct, y_lat, color=color, linewidth=1.8, label=f"{_safe_label(label)}")

    ax_latency.set_title("Total Latency", fontsize=11, fontweight="bold")
    ax_latency.set_xlabel("Percentile")
    ax_latency.set_ylabel("Seconds")
    ax_latency.grid(True, alpha=0.3)
    ax_latency.legend(fontsize=8)

    # ===========================================================
    # Compute averages
    # ===========================================================
    avg_ttft = [np.nanmean(_safe_series(df, "ttft_s")) for df in dfs]
    avg_tbt  = [np.nanmean(_safe_series(df, "avg_tbt_s")) for df in dfs]
    avg_lat  = [np.nanmean(_safe_series(df, "latency_s")) for df in dfs]

    def delta_str(base, compare):
        if np.isnan(base) or np.isnan(compare):
            return "N/A"
        diff = compare - base
        pct = (diff / base * 100) if base > 0 else np.nan
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.3f}s ({sign}{pct:.1f}%)"

    def delta_str2(base, compare):
        if np.isnan(base) or np.isnan(compare):
            return "N/A"
        diff = compare - base
        pct = (diff / base * 100) if base > 0 else np.nan
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.3f}s"

    # ===========================================================
    # TTFT + Avg TBT comparison box (middle-left of first graph)
    # ===========================================================
    try:
        box_ttft_tbt = (
            "Δ Avg TTFT / Avg TBT (vs " + _safe_label(labels[2]) + ")\n"
            f"{_safe_label(labels[0])}: TTFT {delta_str(avg_ttft[2], avg_ttft[0])},  TBT {delta_str2(avg_tbt[2], avg_tbt[0])}\n"
            f"{_safe_label(labels[1])}: TTFT {delta_str(avg_ttft[2], avg_ttft[1])},  TBT {delta_str2(avg_tbt[2], avg_tbt[1])}"
        )
        ax_ttft.text(
            0.03, 0.5, box_ttft_tbt,
            transform=ax_ttft.transAxes,
            fontsize=8.5,
            ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
            family="monospace",
        )
    except Exception as e:
        print(f"[warn] TTFT/TBT box failed: {e}")

    # ===========================================================
    # Total latency comparison box (bottom-right of second graph)
    # ===========================================================
    try:
        box_lat = (
            "Δ Avg Total Latency (vs " + _safe_label(labels[2]) + ")\n"
            f"{_safe_label(labels[0])}: {delta_str(avg_lat[2], avg_lat[0])}\n"
            f"{_safe_label(labels[1])}: {delta_str(avg_lat[2], avg_lat[1])}"
        )
        ax_latency.text(
            0.98, 0.05, box_lat,
            transform=ax_latency.transAxes,
            fontsize=8.5,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8),
            family="monospace",
        )
    except Exception as e:
        print(f"[warn] Latency box failed: {e}")

# ------------------------------------------------------
# Figure 3: GPU Compute Utilization
# ------------------------------------------------------
def plot_gpu_usage_panel(
    ax,
    file1, label1,
    file2, label2,
    file3=None, label3=None,
    smooth_window=1,
    ft_tokens=None,
    colors=None,
):
    def load_log(path):
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["time_rel_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        df = df[df["time_rel_s"] >= 10].reset_index(drop=True)
        df["gpu_util_smooth"] = df["gpu_util"].rolling(window=smooth_window, min_periods=1).mean()
        return df

    # --- Load all logs ---
    df1 = load_log(file1)
    df2 = load_log(file2)
    df3 = load_log(file3) if file3 else None

    # --- Synchronize by shortest timeline ---
    min_end = min(
        df1["time_rel_s"].iloc[-1],
        df2["time_rel_s"].iloc[-1],
        df3["time_rel_s"].iloc[-1] if df3 is not None else float("inf"),
    )

    df1 = df1[df1["time_rel_s"] <= min_end]
    df2 = df2[df2["time_rel_s"] <= min_end]
    if df3 is not None:
        df3 = df3[df3["time_rel_s"] <= min_end]

    # --- Compute averages ---
    avg1, avg2 = df1["gpu_util"].mean(), df2["gpu_util"].mean()
    avg3 = df3["gpu_util"].mean() if df3 is not None else None

    # --- Plot ---
    color_map = colors if colors is not None else ["tab:blue", "tab:orange", "tab:purple"]
    ax.plot(df1["time_rel_s"], df1["gpu_util_smooth"], color=color_map[0], alpha=0.6, linewidth=1.0)
    ax.plot(df2["time_rel_s"], df2["gpu_util_smooth"], color=color_map[1], alpha=0.75, linewidth=1.0)
    if df3 is not None:
        ax.plot(df3["time_rel_s"], df3["gpu_util_smooth"], color=color_map[2], alpha=0.6, linewidth=1.0)

    # --- Avg lines ---
    ax.axhline(avg1, color=color_map[0], linestyle="--", linewidth=2.5, label=f"{label1} avg={avg1:.1f}%")
    ax.axhline(avg2, color=color_map[1], linestyle="--", linewidth=2.5, label=f"{label2} avg={avg2:.1f}%")
    if df3 is not None:
        ax.axhline(avg3, color=color_map[2], linestyle="--", linewidth=2.5, label=f"{label3} avg={avg3:.1f}%")

    # --- Optional fine-tuning tokens info ---
    if ft_tokens is not None:
        ax.plot([], [], " ", label=f"Total trained fine-tuning tokens: {ft_tokens:,}")

    # --- Axis styling ---
    ax.set_title("GPU Compute Utilization", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s, relative)")
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_xlim(0, min_end)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

# ------------------------------------------------------
# Figure 4: Timeline Plot
# ------------------------------------------------------
def plot_timeline_panel(ax, csv_path="timelines/timeline_live.csv"):
    df = pd.read_csv(csv_path)
    df["total_tokens"] = df["prompt_length"] + df["max_new_tokens"]
    df["second_bin"] = np.floor(df["timestamp_s"]).astype(int)

    tokens_per_sec = df.groupby("second_bin")["total_tokens"].sum()
    req_per_sec = df.groupby("second_bin").size()

    # Left axis: total tokens
    ax.plot(tokens_per_sec.index, tokens_per_sec, color="tab:blue", linewidth=1.8, label="Total tokens/sec")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total tokens/sec", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_ylim(top=tokens_per_sec.max() * 1.3)

    # Right axis: requests/sec
    ax2 = ax.twinx()
    ax2.bar(req_per_sec.index, req_per_sec, color="tab:orange", alpha=0.3, width=0.9, label="#Req/sec")
    ax2.set_ylabel("#Requests/sec", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(top=req_per_sec.max() * 1.3)

    ax.set_title("Request Timeline", fontsize=11, fontweight="bold")

# ------------------------------------------------------
# Combine All 4 Panels into One Figure
# ------------------------------------------------------
def plot_all_combined(ft_tokens=None):
    files = ["results/latency_inference.csv", "results/latency_co-serving.csv", "results/latency_slora.csv"]
    labels = ["Inference", "Co-serving", "SLoRA"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    ax_timeline, ax_ttft, ax_latency, ax_gpu = axes

    plot_timeline_panel(ax_timeline, "timelines/timeline_live.csv")
    plot_latency_panels(ax_ttft, ax_latency, files, labels, colors)
    plot_gpu_usage_panel(
        ax_gpu,
        "results/gpu_usage_inference.csv", "inf",
        "results/gpu_usage_co-serving.csv", "co-serving",
        "results/gpu_usage_slora.csv", "SLoRA",
        ft_tokens=ft_tokens,
        colors=colors
    )

    fig.tight_layout()
    out_path = os.path.abspath("results/combined_summary.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"✅ Saved combined 4-panel figure to {out_path}")

if __name__ == "__main__":
    ft_tokens = 6352
    plot_all_combined(ft_tokens)