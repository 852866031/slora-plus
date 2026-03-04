import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Fixed paths ----------------
TIMELINE_CSV = "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/llama3/timeline_live.csv"

# Replaced auto benchmark results path
RESULTS_CSV = "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/llama3/timeline_results.csv"

# Replaced lms.log path with backward log CSV
BWD_LOG_CSV = "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/llama3/bwd_log.csv"

OUT_PATH = "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/llama3/auto_benchmark_summary.png"


# ---------------- Utilities ----------------
def ensure_exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def load_timeline(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["timestamp_s", "prompt_length", "max_new_tokens"]
    if not all(c in df.columns for c in required):
        raise ValueError("timeline_live.csv missing required columns.")
    df = df.sort_values("timestamp_s").reset_index(drop=True)
    df["idx"] = np.arange(len(df))
    return df


def load_results(csv_path: str) -> pd.DataFrame:
    """
    Expected format (from orchestrate_run_timeline.py):
      idx,t_rel_s,latency_s,status,ttft_s,avg_tbt_s,worst_tbt_s
    """
    df = pd.read_csv(csv_path)
    required = ["idx", "latency_s", "status", "ttft_s"]
    if not all(c in df.columns for c in required):
        raise ValueError("timeline_results.csv missing required columns.")

    # status is "ok" on success in the new script
    df["ok"] = df["status"].astype(str).str.strip().eq("ok")
    return df


def parse_bwd_log_csv(csv_path: str):
    """
    Parse backward log CSV:
      timestamp,epoch,batch_idx,batch_tokens,batch_loss,total_processed_tokens

    Returns:
      rel_time_s: np.ndarray
      cum_tokens: np.ndarray (from total_processed_tokens if present)
      avg_tok_s: float
    """
    df = pd.read_csv(csv_path)
    required = ["timestamp", "batch_tokens"]
    if not all(c in df.columns for c in required):
        raise ValueError("bwd_log.csv missing required columns.")

    # Parse timestamps and sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("bwd_log.csv has no valid timestamp rows.")

    # Relative time in seconds
    t0 = df["timestamp"].iloc[0]
    df["rel_time_s"] = (df["timestamp"] - t0).dt.total_seconds()

    # Use total_processed_tokens if available, otherwise cumulative sum of batch_tokens
    if "total_processed_tokens" in df.columns:
        cum_tokens = df["total_processed_tokens"].astype(float).to_numpy()
    else:
        cum_tokens = df["batch_tokens"].astype(float).cumsum().to_numpy()

    rel_time_s = df["rel_time_s"].to_numpy()

    # Compute average tokens/s over the observed interval
    elapsed = float(rel_time_s[-1]) if len(rel_time_s) > 0 else 0.0
    total_tokens = float(cum_tokens[-1]) if len(cum_tokens) > 0 else 0.0
    if elapsed > 0:
        avg_tok_s = total_tokens / elapsed
    else:
        avg_tok_s = float("nan")

    return rel_time_s, cum_tokens, avg_tok_s


# ---------------- Main plotting ----------------
def main():
    ensure_exists(TIMELINE_CSV)
    ensure_exists(RESULTS_CSV)
    ensure_exists(BWD_LOG_CSV)

    timeline = load_timeline(TIMELINE_CSV)
    results = load_results(RESULTS_CSV)

    merged = pd.merge(
        timeline[["idx", "timestamp_s", "prompt_length"]],
        results[["idx", "latency_s", "ttft_s", "ok"]],
        on="idx",
        how="left",
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # -------- Plot 1: Prompt tokens/s vs time --------
    bin_s = 1.0
    t = timeline["timestamp_s"].to_numpy()
    tokens = timeline["prompt_length"].to_numpy().astype(float)

    t0 = float(t.min())
    t1 = float(t.max())

    # Ensure at least one bin even if all timestamps are identical
    nbins = max(1, int(math.ceil((t1 - t0) / bin_s)))
    edges = t0 + np.arange(nbins + 1) * bin_s
    if len(edges) < 2:
        edges = np.array([t0, t0 + bin_s], dtype=float)

    bin_sum, _ = np.histogram(t, bins=edges, weights=tokens)
    centers = edges[:-1] + 0.5 * bin_s
    tok_per_s = bin_sum / bin_s

    ax1.plot(centers, tok_per_s)
    ax1.set_title("Request Timeline (Prompt tokens/s)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Prompt tokens/s")

    # -------- Plot 2: TTFT percentiles --------
    ttft = merged.loc[merged["ok"], "ttft_s"].dropna().astype(float).to_numpy()
    if len(ttft) > 0:
        percentiles = np.array([0, 20, 40, 60, 80, 100])
        values = np.percentile(ttft, percentiles)
        ax2.plot(percentiles, values, marker="o")
    else:
        ax2.text(0.5, 0.5, "No valid TTFT data", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title("TTFT Percentile Curve")
    ax2.set_xlabel("Percentile")
    ax2.set_ylabel("TTFT (s)")

    # -------- Plot 3: E2E latency vs time --------
    ax3.scatter(merged["timestamp_s"], merged["latency_s"], s=10)
    ax3.set_title("Request E2E Latency vs Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Latency (s)")

    # -------- Plot 4: Finetuning tokens over time (from bwd_log.csv) --------
    cum_time, cum_tokens, avg_tok_s = parse_bwd_log_csv(BWD_LOG_CSV)

    ax4.plot(cum_time, cum_tokens)
    ax4.set_title("Finetuning Cumulative Tokens")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Cumulative Tokens")

    avg_text = f"Avg finetune tok/s: {avg_tok_s:.2f}" if np.isfinite(avg_tok_s) else "Avg finetune tok/s: N/A"
    ax4.text(
        0.02,
        0.95,
        avg_text,
        transform=ax4.transAxes,
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=160)
    print(f"[plot_auto] Saved summary figure to {OUT_PATH}")


if __name__ == "__main__":
    main()