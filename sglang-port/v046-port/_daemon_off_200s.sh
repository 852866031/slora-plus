#!/usr/bin/env bash
# Reproduce the 200s co-serving figure (FT + inference) with the MPS daemon OFF,
# so the backward runs UNCAPPED and we can see inference latency degrade.
# Baseline (inference-only) is daemon-independent; only the co-serving pass runs
# with the daemon stopped. Restores the daemon at the end.
set -uo pipefail
cd "$(dirname "$0")"
PY=/mnt/weka/home/jianshu.she/miniconda3/bin/python
PORT=30260
O=output
mkdir -p "$O" plots
export CUDA_VISIBLE_DEVICES=0

echo "########## [1/5] inference-only baseline (daemon state irrelevant) ##########"
$PY auto_benchmark_sglang.py --tight --timeline-gpu CUSTOM200 --no-radix \
    --port $PORT > /tmp/d200_baseline.log 2>&1
cp "$O/timeline_results_tight_inf.csv" "$O/slora_200s_nodaemon_src.csv" 2>/dev/null || echo "WARN: no inf csv"

echo "########## [2/5] STOP MPS daemon ##########"
echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 2
echo "daemon procs now: $(ps aux | grep -c '[n]vidia-cuda-mps-control')"

echo "########## [3/5] co-serving pass, DAEMON OFF (backward uncapped) ##########"
SGLANG_DS_RPS_THROTTLE=1 SGLANG_DS_RPS_CLOSE=10 SGLANG_DS_RPS_OPEN=6 SGLANG_DS_RPS_WINDOW_S=2 \
$PY auto_benchmark_sglang.py --tight --timeline-gpu CUSTOM200 \
    --co --store-driven --real-backward --backward-subprocess --backward-mps-pct 10 \
    --port $PORT > /tmp/d200_coserve.log 2>&1
cp "$O/timeline_results_tight_co_real_sub.csv" "$O/dserve_200s_nodaemon_src.csv" 2>/dev/null || echo "WARN: no co csv"
cp "$O/server_tight_co.log" "$O/server_200s_nodaemon.log" 2>/dev/null

echo "########## [4/5] RESTART MPS daemon ##########"
nvidia-cuda-mps-control -d 2>/dev/null; sleep 2
echo "daemon procs now: $(ps aux | grep -c '[n]vidia-cuda-mps-control')"

echo "########## [5/5] adapt + plot ##########"
$PY make_compare_inputs.py --results-in "$O/slora_200s_nodaemon_src.csv" \
    --results-out "$O/slora_200s_nodaemon.csv"
$PY make_compare_inputs.py --results-in "$O/dserve_200s_nodaemon_src.csv" \
    --results-out "$O/dserve_200s_nodaemon.csv" \
    --server-log "$O/server_200s_nodaemon.log" --bwd-out "$O/bwd_200s_nodaemon.csv"

$PY compare_slora_dserve.py \
    --slora "$O/slora_200s_nodaemon.csv" --dserve "$O/dserve_200s_nodaemon.csv" \
    --bwd "$O/bwd_200s_nodaemon.csv" \
    --timeline ../../eval/llama3/timelines/CUSTOM200/timeline_tight.csv \
    --gpu-name "H200 — MPS daemon OFF (backward uncapped)" \
    --output plots/compare_slora_dserve_200s_nodaemon.png

echo "########## DONE -> plots/compare_slora_dserve_200s_nodaemon.png ##########"
# quick latency summary
$PY - <<'PYEOF'
import csv
def mean_lat(p):
    try:
        r=[float(x["latency_s"]) for x in csv.DictReader(open(p)) if x.get("latency_s") and (x.get("status","ok")!="error")]
        return sum(r)/len(r), len(r)
    except Exception as e: return None,0
for tag,p in [("baseline(inf-only)","output/slora_200s_nodaemon.csv"),("co-serve DAEMON OFF","output/dserve_200s_nodaemon.csv")]:
    m,n=mean_lat(p); print(f"{tag}: mean E2E={m:.3f}s over {n} reqs" if m else f"{tag}: n/a")
PYEOF
