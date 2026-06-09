"""plot_mps_daemon.py — show the MPS daemon is load-bearing for the 10% cap.

Bars: pure-FT throughput (8B, rank-16 LoRA, H200) at the SAME --backward-mps-pct=10
setting, with the MPS control daemon ON vs OFF, plus the uncapped (100%) reference.
Without the daemon, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10 is inert -> the backward
grabs the whole GPU -> throughput collapses back to the uncapped number.

Usage: python plot_mps_daemon.py <on10> <off10> <ref100>
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

on10 = float(sys.argv[1]) if len(sys.argv) > 1 else 422.0
off10 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
ref100 = float(sys.argv[3]) if len(sys.argv) > 3 else 881.0

labels = ["10% cap\ndaemon ON", "10% cap\ndaemon OFF", "100% (uncapped)\ndaemon ON\n(reference)"]
vals = [on10, off10, ref100]
colors = ["#2e7d32", "#c62828", "#9e9e9e"]

fig, ax = plt.subplots(figsize=(7.2, 5.2))
bars = ax.bar(labels, vals, color=colors, width=0.6, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 12, f"{v:.0f}", ha="center",
            va="bottom", fontsize=12, fontweight="bold")

ax.axhline(ref100, ls="--", lw=1.0, color="#9e9e9e", alpha=0.8)
ax.set_ylabel("Pure-FT throughput (tok/s)  ·  8B rank-16, H200", fontsize=11)
ax.set_title("The MPS daemon is what makes the 10% cap real\n"
             "Same --backward-mps-pct=10; daemon OFF → cap inert → backward eats the whole GPU",
             fontsize=11.5)
ax.set_ylim(0, max(vals) * 1.18)
ax.grid(axis="y", ls=":", alpha=0.4)

# annotate the collapse
if off10 > 0:
    ax.annotate(f"+{(off10/on10 - 1)*100:.0f}% — cap gone",
                xy=(1, off10), xytext=(1, off10 + 90),
                ha="center", fontsize=10, color="#c62828",
                arrowprops=dict(arrowstyle="->", color="#c62828"))

fig.tight_layout()
out = "plots/mps_daemon_effect.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}  (on10={on10} off10={off10} ref100={ref100})")
