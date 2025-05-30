import json
import matplotlib.pyplot as plt

# Initialize lists
num_adapters = []
avg_latency = []
throughput = []

# Read data from JSONL file
with open('results.jsonl', 'r') as f:
    for line in f:
        record = json.loads(line)
        config = record["config"]
        result = record["result"]
        num_adapters.append(config["num_adapters"])
        avg_latency.append(result["avg_latency"])
        throughput.append(result["throughput"])

# Plot average latency
plt.figure()
plt.plot(num_adapters, avg_latency, marker='o')
plt.xlabel("Number of Adapters")
plt.ylabel("Average Latency (s)")
plt.title("Number of Adapters vs Average Latency")
plt.grid(True)
plt.savefig("num_adapters_vs_avg_latency.png")

# Plot throughput
plt.figure()
plt.plot(num_adapters, throughput, marker='o')
plt.xlabel("Number of Adapters")
plt.ylabel("Throughput (req/s)")
plt.title("Number of Adapters vs Throughput")
plt.grid(True)
plt.savefig("num_adapters_vs_throughput.png")