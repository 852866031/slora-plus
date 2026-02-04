# nutanix_realworld_summary.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# ------------------------------------------------------------
# Plot Configuration (copied from main script and simplified)
# ------------------------------------------------------------
PLOT_CFG = {
    "timeline": {
        "left_ylim":  [0, 1900],
        "left_ytick": [0, 500, 1000, 1500],
        "right_ylim": [0, 20],
        "right_ytick": [0, 5, 10, 15, 20],
    },
    "gpu": {
        "left_ylim":  [0, 120],
        "left_ytick": [0, 25, 50, 75, 100],
    },
    "x_tick_pad": -5,
    "y_tick_pad": -5,
}

# ------------------------------------------------------------
# Style
# ------------------------------------------------------------
def set_paper_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.4,
        "grid.color": "#dddddd",
        "lines.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


# ------------------------------------------------------------
# Timeline plot (copied from your big file, simplified)
# ------------------------------------------------------------
def plot_timeline(ax, csv_path):
    cfg = PLOT_CFG["timeline"]

    df = pd.read_csv(csv_path)
    df["total_tokens"] = df["prompt_length"] + df["max_new_tokens"]
    df["second_bin"] = np.floor(df["timestamp_s"]).astype(int)

    start = int(df["second_bin"].min())
    end   = int(df["second_bin"].max())
    full_index = range(start, end + 1)

    tokens_raw = df.groupby("second_bin")["total_tokens"].sum()
    rps_raw    = df.groupby("second_bin").size()

    tokens = tokens_raw.reindex(full_index, fill_value=0)
    rps    = rps_raw.reindex(full_index, fill_value=0)

    x_vals = np.array(list(full_index))

    ax.plot(x_vals, tokens, color="#5E4FA2", linewidth=1.0, label="Total Inference Tokens/sec")
    ax.set_ylabel("Tokens/sec", color="#5E4FA2", fontsize=10)
    ax.tick_params(axis="y", labelcolor="#5E4FA2")

    ax2 = ax.twinx()
    ax2.plot(x_vals, rps, color="#000000", alpha=0.8, linewidth=0.8, label="Requests/sec")
    ax2.fill_between(x_vals, rps, color="#000000", alpha=0.15)

    ax2.set_ylabel("Requests/sec", color="#000000", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#000000")
    ax.set_xticks([0, 400, 800, 1200])

    ax.set_ylim(cfg["left_ylim"])
    ax.set_yticks(cfg["left_ytick"])
    ax2.set_ylim(cfg["right_ylim"])
    ax2.set_yticks(cfg["right_ytick"])
    ax.tick_params(axis="x", pad=0.3, grid_alpha=0)
    ax.tick_params(axis="y", pad=0.3, grid_alpha=0)
    ax2.tick_params(axis="y", pad=-1)
    ax2.tick_params(axis="x", pad=-1)
    ax.tick_params(axis="y", direction="out", length=0)
    ax2.tick_params(axis="y", direction="out", length=0)

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_title("Company X Real-world Inference Workload Trace", fontsize=12)

    return ax2


# ------------------------------------------------------------
# GPU Utilization [Simple version]
# ------------------------------------------------------------
def plot_gpu_usage(ax, csv_path, threshold=80, pre_seconds=2.0):
    cfg = PLOT_CFG["gpu"]

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["time_rel_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df["gpu_util_smooth"] = df["gpu_util"].rolling(window=3, min_periods=1).mean()

    # Main GPU util line
    ax.plot(
        df["time_rel_s"], df["gpu_util_smooth"],
        color="#1F77B4", linewidth=1.0, label="GPU Utilization"
    )

    # Compute average and draw horizontal line
    avg_util = df["gpu_util"].mean()
    ax.axhline(
        avg_util,
        color="#1F77B4",
        linestyle="--",
        linewidth=1.0,
        alpha=0.9
    )

    # Annotate average utilization **on the line**
        # Annotate average utilization on the right side
    ax.text(
        1.01,                                # x: just outside the axis on the right
        avg_util,                            # y: SAME data value as the dashed line
        f"{avg_util:.1f}%",
        transform=ax.get_yaxis_transform(),  # x in axes coords, y in data coords
        ha="left",
        va="center",
        fontsize=7,
        color="#1F77B4",
        bbox=dict(
            boxstyle="round,pad=0.20",
            fc="white",
            ec="none",
            alpha=0.8,
        ),
    )

    # Formatting
    ax.set_ylim(cfg["left_ylim"])
    ax.set_yticks(cfg["left_ytick"])
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("GPU Utilization (%)", fontsize=10)
    ax.tick_params(axis="x", pad=-4, grid_alpha=0)
    ax.tick_params(axis="y", pad=-4, grid_alpha=0)
    ax.set_xticks([0, 400, 800, 1200])
    ax.grid(alpha=0.3)
    ax.set_title("S-LoRA GPU Utilization Under Real-world Workload", fontsize=12)

    return avg_util

# ------------------------------------------------------------
# Combined single-column figure
# ------------------------------------------------------------
def plot_nutanix_realworld(nutanix_dir):
    """
    nutanix_dir should contain:
      - timeline_live.csv
      - gpu_usage_slora.csv
    """

    timeline_csv = os.path.join(nutanix_dir, "timeline_live.csv")
    gpu_csv      = os.path.join(nutanix_dir, "gpu_usage_slora.csv")

    set_paper_style()

    fig, axes = plt.subplots(
        2, 1,
        figsize=(6, 4),
        gridspec_kw={"height_ratios": [1.2, 1]}
    )
    
    # Row 1: workload trace
    plot_timeline(axes[0], timeline_csv)

    # Row 2: GPU trace
    plot_gpu_usage(axes[1], gpu_csv)

    plt.tight_layout()

    out_path = os.path.abspath("nutanix_summary.pdf")
    plt.savefig(
        out_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.0
        )
    plt.close(fig)

    print(f"✅ Saved Nutanix real-world workload figure → {out_path}")


# ------------------------------------------------------------
if __name__ == "__main__":
    # Change this if your directory name differs
    plot_nutanix_realworld("nutanix")