import pandas as pd
import matplotlib.pyplot as plt

def plot_two_latency_files(file1, label1, file2, label2, out_png="compare_latency.png"):
    # Load both CSVs
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Only keep successful requests
    df1_ok = df1[df1["status"] == "ok"]
    df2_ok = df2[df2["status"] == "ok"]

    plt.figure(figsize=(12, 5))

    # Scatter for each file
    plt.scatter(df1_ok["t_rel_s"], df1_ok["latency_s"],
                s=12, alpha=0.6, label=label1)
    plt.scatter(df2_ok["t_rel_s"], df2_ok["latency_s"],
                s=12, alpha=0.6, label=label2)

    # Moving average smoothing
    def moving_avg(x, y, span):
        ma_x, ma_y = [], []
        acc = 0.0
        for i, val in enumerate(y):
            acc += val
            if i >= span:
                acc -= y[i - span]
            if i >= span - 1:
                ma_x.append(x[i])
                ma_y.append(acc / span)
        return ma_x, ma_y

    for df, lbl, color in [(df1_ok, label1, "tab:blue"), (df2_ok, label2, "tab:orange")]:
        if not df.empty:
            span = max(3, min(200, int(len(df) / (df["t_rel_s"].max() - df["t_rel_s"].min() + 1e-9))))
            if span >= 3:
                ma_x, ma_y = moving_avg(df["t_rel_s"].to_numpy(), df["latency_s"].to_numpy(), span)
                plt.plot(ma_x, ma_y, linewidth=2.0, label=f"{lbl} (avg)", color=color)

    # Compute and annotate average difference
    avg1 = df1_ok["latency_s"].mean()
    avg2 = df2_ok["latency_s"].mean()
    diff = avg2 - avg1

    annotation = f"Avg difference: {diff*1000:.1f} ms ({label2} - {label1})"
    plt.annotate(annotation,
                 xy=(0.5, 0.95), xycoords="axes fraction",
                 ha="center", fontsize=12, bbox=dict(boxstyle="round", fc="w", alpha=0.6))

    plt.xlabel("Time since start (s)")
    plt.ylabel("Latency (s)")
    plt.title("Latency timeline comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"✅ Saved comparison plot to {out_png}")
    plt.show()

plot_two_latency_files("rps5_dur3s_latency_inf_no_saving.csv", "inf",
                      "rps5_dur3s_latency_co_saving.csv", "co-serve")