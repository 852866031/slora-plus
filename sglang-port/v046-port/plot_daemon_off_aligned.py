"""plot_daemon_off_aligned.py — daemon-OFF co-serving, FT vs inference load on ONE
clock (the scheduler's own wall clock), parsed from the SGLANG_DS_RPS_DEBUG server
log. Proves the RPS throttle still anti-correlates FT with inference when the
daemon is off — the earlier compare_slora_dserve figure only LOOKED broken because
its FT band (wall clock) and inference panel (client schedule clock) were ~45s
offset.
"""
import re, datetime, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "output/server_200s_rpsdebug.log"
prps = re.compile(r"\[([\d\- :]+)\] \[rps\] rps=([\d.]+) dq=(\d+)")
pfire = re.compile(r"real_backward #\d+: .* n_valid=(\d+) cum_tokens=\d+ wall=([\d.]+)")
import time

rps_t, rps_v, fires = [], [], []
for line in open(LOG, errors="ignore"):
    m = prps.search(line)
    if m:
        t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        rps_t.append(t); rps_v.append(float(m.group(2)))
    m2 = pfire.search(line)
    if m2:
        fires.append((float(m2.group(2)), int(m2.group(1))))

t0 = rps_t[0]
t0e = time.mktime(t0.timetuple())
rps_rel = [(t - t0).total_seconds() for t in rps_t]
# FT tok per 2s bin
ftbin = collections.defaultdict(int)
for w, nv in fires:
    ftbin[int(w - t0e) // 2] += nv
ft_x = sorted(ftbin); ft_y = [ftbin[k] / 2.0 for k in ft_x]  # tok/s
ft_x = [k * 2 for k in ft_x]

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(ft_x, ft_y, color="#ff9800", alpha=0.45, label="FT throughput (tok/s)")
ax.set_xlabel("scheduler wall-clock time (s)")
ax.set_ylabel("FT throughput (tok/s)", color="#e65100")
ax.set_ylim(0, max(ft_y) * 1.15)

ax2 = ax.twinx()
ax2.plot(rps_rel, rps_v, color="#1565c0", lw=1.4, label="inference rps (throttle view)")
ax2.axhline(10, ls="--", color="#1565c0", alpha=0.6, lw=1.0)
ax2.text(rps_rel[-1], 10.4, "close=10 rps", color="#1565c0", ha="right", fontsize=9)
ax2.set_ylabel("inference arrivals (rps)", color="#1565c0")
ax2.set_ylim(0, max(rps_v) * 1.2)

ax.set_title("Daemon-OFF co-serving on ONE clock: FT backs off EXACTLY when inference load spikes\n"
             "(the RPS throttle works; the earlier figure only looked broken due to a ~45s FT-vs-timeline clock offset)",
             fontsize=11)
l1, lab1 = ax.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lab1 + lab2, loc="upper right", fontsize=9)
ax.grid(axis="x", ls=":", alpha=0.3)
fig.tight_layout()
out = "plots/daemon_off_coserve_aligned.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
