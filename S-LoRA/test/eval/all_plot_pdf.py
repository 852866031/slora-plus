# multi_experiment_summary.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ------------------------------------------------------------
# Aesthetic settings for paper-quality figures
# ------------------------------------------------------------
def set_paper_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.color": "#dddddd",
        "lines.linewidth": 1.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.transparent": True,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def _safe_series(df, name):
    return df[name].dropna() if name in df.columns else pd.Series([], dtype=float)

def _percentile_curve(series):
    vals = series.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([]), np.array([])
    pct = np.linspace(0, 100, 101)
    return pct, np.percentile(vals, pct)


# ------------------------------------------------------------
# Per-panel plotters
# ------------------------------------------------------------
def plot_timeline_panel(ax, csv_path):
    df = pd.read_csv(csv_path)
    df["total_tokens"] = df["prompt_length"] + df["max_new_tokens"]
    df["second_bin"] = np.floor(df["timestamp_s"]).astype(int)
    tokens_per_sec = df.groupby("second_bin")["total_tokens"].sum()
    req_per_sec = df.groupby("second_bin").size()

    ax.plot(tokens_per_sec.index, tokens_per_sec, color="tab:blue", linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total tokens/sec", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax.twinx()
    ax2.bar(req_per_sec.index, req_per_sec, color="tab:orange", alpha=0.25, width=0.9)
    ax2.set_ylabel("#Requests/sec", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, req_per_sec.max() * 1.2)
    ax.set_title("Request Timeline", pad=2)


def plot_latency_panels(ax_ttft, ax_latency, files, labels, colors):
    dfs = [pd.read_csv(f) for f in files]
    dfs = [df[df.get("status", "ok") == "ok"] for df in dfs]
    pct = np.linspace(0, 100, 101)

    # TTFT / Avg TBT
    for df, label, color in zip(dfs, labels, colors):
        x, y_ttft = _percentile_curve(_safe_series(df, "ttft_s"))
        _, y_tbt = _percentile_curve(_safe_series(df, "avg_tbt_s"))
        if y_ttft.size:
            ax_ttft.plot(x, y_ttft, color=color, linestyle="-", label=f"TTFT {label}")
        if y_tbt.size:
            ax_ttft.plot(x, y_tbt, color=color, linestyle="--", label=f"TBT {label}")
    ax_ttft.set_title("TTFT / Avg TBT", pad=2)
    ax_ttft.set_xlabel("Percentile")
    ax_ttft.set_ylabel("Seconds")
    ax_ttft.grid(True, alpha=0.3)
    ax_ttft.legend(fontsize=6, loc="upper left", ncol=1, frameon=False)

    # Total Latency
    for df, label, color in zip(dfs, labels, colors):
        x, y_lat = _percentile_curve(_safe_series(df, "latency_s"))
        if y_lat.size:
            ax_latency.plot(x, y_lat, color=color, linewidth=1.0, label=label)
    ax_latency.set_title("Total Latency", pad=2)
    ax_latency.set_xlabel("Percentile")
    ax_latency.set_ylabel("Seconds")
    ax_latency.legend(fontsize=6, loc="upper left", frameon=False)
    ax_latency.grid(True, alpha=0.3)


def plot_gpu_usage_panel(ax, files, labels, colors):
    def load_log(path):
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["time_rel_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        df["gpu_util_smooth"] = df["gpu_util"].rolling(window=3, min_periods=1).mean()
        return df

    dfs = [load_log(f) for f in files]
    min_end = min(d["time_rel_s"].iloc[-1] for d in dfs)
    dfs = [d[d["time_rel_s"] <= min_end] for d in dfs]

    for df, label, color in zip(dfs, labels, colors):
        ax.plot(df["time_rel_s"], df["gpu_util_smooth"], color=color, linewidth=1.0, alpha=0.9)
        avg = df["gpu_util"].mean()
        ax.axhline(avg, color=color, linestyle="--", linewidth=1.0, alpha=0.8, label=f"{label} {avg:.1f}%")

    ax.set_title("GPU Utilization", pad=2)
    ax.set_xlabel("Time (s, relative)")
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)


# ------------------------------------------------------------
# Main combined plot (4×3)
# ------------------------------------------------------------
def plot_multi_experiment_summary(base_dirs):
    set_paper_style()

    n_rows, n_cols = 4, len(base_dirs)
    fig = plt.figure(figsize=(7.5, 6.8))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.3, hspace=0.4)

    colors = sns.color_palette("colorblind", n_colors=3)
    labels = ["Inference", "Co-serving", "SLoRA"]

    for col, base in enumerate(base_dirs):
        # Each directory (tight/loose/nutanix)
        lat_files = [f"{base}/latency_inference.csv", f"{base}/latency_co-serving.csv", f"{base}/latency_slora.csv"]
        gpu_files = [f"{base}/gpu_usage_inference.csv", f"{base}/gpu_usage_co-serving.csv", f"{base}/gpu_usage_slora.csv"]
        timeline_csv = f"{base}/timeline_live.csv"

        # Row 1: timeline
        ax_timeline = fig.add_subplot(gs[0, col])
        plot_timeline_panel(ax_timeline, timeline_csv)
        if col > 0:
            ax_timeline.set_ylabel("")

        # Row 2: TTFT/TBT
        ax_ttft = fig.add_subplot(gs[1, col])
        plot_latency_panels(ax_ttft, ax_ttft, lat_files, labels, colors)
        if col > 0:
            ax_ttft.set_ylabel("")

        # Row 3: total latency
        ax_lat = fig.add_subplot(gs[2, col])
        plot_latency_panels(ax_lat, ax_lat, lat_files, labels, colors)
        if col > 0:
            ax_lat.set_ylabel("")

        # Row 4: GPU usage
        ax_gpu = fig.add_subplot(gs[3, col])
        plot_gpu_usage_panel(ax_gpu, gpu_files, labels, colors)
        if col > 0:
            ax_gpu.set_ylabel("")

        # Column title
        ax_timeline.set_title(os.path.basename(base).capitalize(), fontsize=9, pad=8)

    # Global labels
    fig.text(0.5, 0.04, "Time / Percentile", ha="center", fontsize=8)
    fig.text(0.02, 0.5, "Metric Value", va="center", rotation="vertical", fontsize=8)

    out_path = os.path.abspath("results/comparison_summary.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved multi-experiment summary to {out_path}")


if __name__ == "__main__":
    # Adjust order or add/remove dirs as needed
    dirs = ["co_results/loose", "co_results/nutanix", "co_results/tight"]
    plot_multi_experiment_summary(dirs)