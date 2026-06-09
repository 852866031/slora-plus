#!/usr/bin/env bash
# 20-minute co-serving deliverable (daemon ON, 1B) — same pipeline + fixes as the
# 200s one: aligned FT band (anchor-trim), smoothed, latency outliers clipped.
#   3 CSVs -> output/{slora,dserve,bwd}_20m.csv
#   figure -> plots/compare_slora_dserve_20m.png
set -uo pipefail
cd "$(dirname "$0")"
PY=/mnt/weka/home/jianshu.she/miniconda3/bin/python
PORT=30263
O=output
TL=../../eval/llama3/timelines/CUSTOM20M/timeline_tight.csv
mkdir -p "$O" plots
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps

echo "## daemon check ##"; ps aux | grep -c '[n]vidia-cuda-mps-control'

echo "## [1/4] inference-only baseline (20 min) ##"
$PY auto_benchmark_sglang.py --tight --timeline-gpu CUSTOM20M --no-radix \
    --port $PORT > /tmp/m20_baseline.log 2>&1
cp "$O/timeline_results_tight_inf.csv" "$O/_inf20_src.csv"

echo "## [2/4] co-serving (daemon ON, backward capped 10%) (20 min) ##"
SGLANG_DS_RPS_THROTTLE=1 SGLANG_DS_RPS_CLOSE=10 SGLANG_DS_RPS_OPEN=6 SGLANG_DS_RPS_WINDOW_S=2 \
$PY auto_benchmark_sglang.py --tight --timeline-gpu CUSTOM20M \
    --co --store-driven --real-backward --backward-subprocess --backward-mps-pct 10 \
    --port $PORT > /tmp/m20_coserve.log 2>&1
cp "$O/timeline_results_tight_co_real_sub.csv" "$O/_co20_src.csv"
cp "$O/server_tight_co.log" "$O/_co20_server.log"

echo "## [3/4] adapt -> 3 CSVs (FT trimmed to timeline window) ##"
ANCHOR=$(grep -oE "timeline_anchor_wall=[0-9.]+" /tmp/m20_coserve.log | head -1 | cut -d= -f2)
echo "timeline_anchor_wall=$ANCHOR"
$PY make_compare_inputs.py --results-in "$O/_inf20_src.csv" --results-out "$O/slora_20m.csv"
$PY make_compare_inputs.py --results-in "$O/_co20_src.csv"  --results-out "$O/dserve_20m.csv" \
    --server-log "$O/_co20_server.log" --bwd-out "$O/bwd_20m.csv" \
    ${ANCHOR:+--anchor-wall "$ANCHOR" --tl-span 1200}

echo "## [4/4] figure (smoothed 20s, lat-clip 0.35) ##"
$PY compare_slora_dserve.py \
    --slora "$O/slora_20m.csv" --dserve "$O/dserve_20m.csv" --bwd "$O/bwd_20m.csv" \
    --timeline "$TL" --slora-latency-factor 1.0 --window 20 --lat-clip 0.35 \
    --gpu-name "H200 — co-serving 20min (daemon ON)" \
    --output plots/compare_slora_dserve_20m.png

echo "## latency summary ##"
$PY - <<'PYEOF'
import csv,statistics as st
def m(p):
    r=[float(x["latency_s"]) for x in csv.DictReader(open(p)) if x.get("latency_s") and (x.get("status","ok")!="error")]
    return st.mean(r),len(r)
for tag,p in [("baseline","output/slora_20m.csv"),("co-serve","output/dserve_20m.csv")]:
    a,n=m(p); print(f"{tag}: mean E2E={a:.3f}s n={n}")
fr=list(csv.DictReader(open("output/bwd_20m.csv"))); tot=sum(int(float(x["batch_tokens"])) for x in fr)
print(f"FT: {len(fr)} fires, {tot} tok")
PYEOF
echo "## DONE ##"
