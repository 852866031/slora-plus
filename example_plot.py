import matplotlib.pyplot as plt
import numpy as np

# X-axis: proportion of inference requests
x = np.linspace(0, 1, 100)

# Inference latency (left y-axis)
latency_ours = 32 + 10 * np.power(x, 1.5)  # Slightly higher at start, increases gently
latency_baseline = 30 + 30 * np.power(x, 3)

# Fine-tuning throughput (right y-axis)
ft_ours = 100 * (1 - x)               # Decreases from 100 to 0
ft_baseline = np.full_like(x, 50)     # Flat line at 50

fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot inference latency
ax1.set_xlabel("Inference Request Ratio")
ax1.set_ylabel("Inference Latency (ms)", color='tab:red')
ax1.plot(x, latency_baseline, label='Baseline Inference Latency (2inf)', color='tab:red', linestyle='-')
ax1.plot(x, latency_ours, label='Ours Inference Latency', color='tab:red', linestyle='--')
ax1.tick_params(axis='y', labelcolor='tab:red')

# Second y-axis for fine-tuning throughput
ax2 = ax1.twinx()
ax2.set_ylabel("Fine-tuning Throughput (samples/sec)", color='tab:blue')
ax2.plot(x, ft_baseline, label='Baseline FT Throughput (2ft)', color='tab:blue', linestyle='-')
ax2.plot(x, ft_ours, label='Ours FT Throughput', color='tab:blue', linestyle='--')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
plt.legend(lines_1 + lines_2, labels_1 + labels_2)

plt.title("Inference vs. Fine-tuning Performance Under Varying Inference Load")
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("co_serving_plot_adjusted.png", dpi=300)
plt.close()