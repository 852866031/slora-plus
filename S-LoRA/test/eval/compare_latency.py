import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_two_latency_files(file1, label1, file2, label2, out_png="compare_latency.png"):
    # Load & filter
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    df1_ok = df1[df1["status"] == "ok"].copy()
    df2_ok = df2[df2["status"] == "ok"].copy()

    lat1 = df1_ok["latency_s"].to_numpy()
    lat2 = df2_ok["latency_s"].to_numpy()

    # Percentile grid (0..100)
    pct = np.linspace(0, 100, 101)
    y1 = np.percentile(lat1, pct) if lat1.size else np.array([])
    y2 = np.percentile(lat2, pct) if lat2.size else np.array([])

    # --- Duration & RPS per your definition (baseline = label2) ---
    def compute_duration_and_rps(df):
        if "t_rel_s" not in df.columns or df.empty:
            return np.nan, np.nan, np.nan, np.nan
        t_first = float(df["t_rel_s"].min())
        t_last  = float(df["t_rel_s"].max())
        arrival_span = max(0.0, t_last - t_first)
        duration = math.ceil(arrival_span)  # your round_up()
        total_reqs = int(len(df))           # total rows (ok + err)
        ok_reqs = int((df["status"] == "ok").sum()) if "status" in df.columns else total_reqs
        rps = (total_reqs / duration) if duration > 0 else np.nan
        return duration, rps, total_reqs, ok_reqs

    duration_s, eff_rps, total_reqs_baseline, ok_reqs_baseline = compute_duration_and_rps(df2)

    # --- Mean diff & overhead vs baseline
    avg1 = float(np.mean(lat1)) if lat1.size else np.nan
    avg2 = float(np.mean(lat2)) if lat2.size else np.nan
    diff = (avg1 - avg2) if (np.isfinite(avg1) and np.isfinite(avg2)) else np.nan
    overhead_str = "n/a"
    if np.isfinite(avg2) and avg2 > 0 and np.isfinite(diff):
        overhead_str = f"{(diff / avg2) * 100.0:+.1f}%"

    # --- Plot: Percentile → Latency
    plt.figure(figsize=(10, 5))
    if y1.size:
        plt.plot(pct, y1, label=label1)
    if y2.size:
        plt.plot(pct, y2, label=label2)

    for p in (50, 90, 99):
        plt.axvline(p, linestyle=":", linewidth=1)

    # Annotation (bottom-right, no delta-P)
    lines = []
    if np.isfinite(duration_s):
        lines.append(f"Baseline {label2} window: {duration_s:.0f}s")
    if np.isfinite(eff_rps):
        lines.append(f"Effective RPS: {eff_rps:.2f}")
    lines.append(f"#Requests: {total_reqs_baseline}")
    if np.isfinite(diff):
        lines.append(f"Mean diff: {diff*1000:.1f} ms ({label1} - {label2})")
        lines.append(f"Overhead vs {label2}: {overhead_str}")

    if lines:
        plt.annotate(
            "\n".join(lines),
            xy=(0.98, 0.02), xycoords="axes fraction",
            ha="right", va="bottom",
            fontsize=11, bbox=dict(boxstyle="round", fc="w", alpha=0.7)
        )

    plt.xlabel("Percentile")
    plt.ylabel("Latency (s)")
    plt.title("Latency percentile comparison (lower is faster)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"✅ Saved comparison plot to {out_png}")
# Example:
plot_two_latency_files("latency_co-serving.csv", "co-serve", "latency_inference.csv", "inf")