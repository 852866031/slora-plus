#!/usr/bin/env bash
# Production 200s co-serving deliverable (daemon ON = the real config):
#   3 CSVs  -> output/{slora,dserve,bwd}_200s.csv
#   figure  -> plots/compare_slora_dserve_200s.png
# slora  = inference-only baseline latency
# dserve = co-serving latency
# bwd    = FT (backward) throughput
set -uo pipefail
cd "$(dirname "$0")"
PY=/mnt/weka/home/jianshu.she/miniconda3/bin/python
PORT=30262
O=output
mkdir -p "$O" plots
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps

echo "## daemon check ##"; ps aux | grep -c '[n]vidia-cuda-mps-control'

echo "## [1/4] inference-only baseline ##"
$PY auto_benchmark_sglang.py --tight --timeline-gpu CUSTOM200 --no-radix \
    --port $PORT > /tmp/prod200_baseline.log 2>&1
cp "$O/timeline_results_tight_inf.csv" "$O/_inf_src.csv"

echo "## [2/4] co-serving (daemon ON, backward capped 10%) ##"
SGLANG_DS_RPS_THROTTLE=1 SGLANG_DS_RPS_CLOSE=10 SGLANG_DS_RPS_OPEN=6 SGLANG_DS_RPS_WINDOW_S=2 \
$PY auto_benchmark_sglang.py --tight --timeline-gpu CUSTOM200 \
    --co --store-driven --real-backward --backward-subprocess --backward-mps-pct 10 \
    --port $PORT > /tmp/prod200_coserve.log 2>&1
cp "$O/timeline_results_tight_co_real_sub.csv" "$O/_co_src.csv"
cp "$O/server_tight_co.log" "$O/_co_server.log"

echo "## [3/4] adapt -> the 3 deliverable CSVs ##"
# Trim FT fires to the timeline window so the FT band aligns to the inference
# clock (FT starts at gate-open ~40s before the timeline drives; without this
# the band is skewed ~40s — the alignment bug the user caught). The benchmark
# logs the absolute timeline start as `timeline_anchor_wall=`.
ANCHOR=$(grep -oE "timeline_anchor_wall=[0-9.]+" /tmp/prod200_coserve.log | head -1 | cut -d= -f2)
echo "timeline_anchor_wall=$ANCHOR"
$PY make_compare_inputs.py --results-in "$O/_inf_src.csv" --results-out "$O/slora_200s.csv"
$PY make_compare_inputs.py --results-in "$O/_co_src.csv"  --results-out "$O/dserve_200s.csv" \
    --server-log "$O/_co_server.log" --bwd-out "$O/bwd_200s.csv" \
    ${ANCHOR:+--anchor-wall "$ANCHOR" --tl-span 200}

echo "## [4/4] complete figure ##"
$PY compare_slora_dserve.py \
    --slora "$O/slora_200s.csv" --dserve "$O/dserve_200s.csv" --bwd "$O/bwd_200s.csv" \
    --timeline ../../eval/llama3/timelines/CUSTOM200/timeline_tight.csv \
    --slora-latency-factor 1.0 --gpu-name "H200 — co-serving (daemon ON)" \
    --output plots/compare_slora_dserve_200s.png

echo "## latency summary ##"
$PY - <<'PYEOF'
import csv,statistics as st
def m(p):
    r=[float(x["latency_s"]) for x in csv.DictReader(open(p)) if x.get("latency_s") and (x.get("status","ok")!="error")]
    return st.mean(r),len(r)
for tag,p in [("baseline","output/slora_200s.csv"),("co-serve","output/dserve_200s.csv")]:
    a,n=m(p); print(f"{tag}: mean E2E={a:.3f}s n={n}")
import csv
fr=list(csv.DictReader(open("output/bwd_200s.csv")))
tot=sum(int(float(x["batch_tokens"])) for x in fr)
print(f"FT: {len(fr)} fires, {tot} tok")
PYEOF
echo "## DONE ##"
