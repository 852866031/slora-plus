#!/usr/bin/env bash
# Re-run ONLY the daemon-OFF co-serve with RPS-throttle debug logging, to see
# what rps the throttle measured during the burst + scheduler tick interval.
set -uo pipefail
cd "$(dirname "$0")"
PY=/mnt/weka/home/jianshu.she/miniconda3/bin/python
PORT=30261
export CUDA_VISIBLE_DEVICES=0

echo "## STOP daemon ##"
echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 2
echo "daemon procs: $(ps aux | grep -c '[n]vidia-cuda-mps-control')"

echo "## co-serve DAEMON OFF + RPS_DEBUG ##"
SGLANG_DS_RPS_DEBUG=1 SGLANG_DS_RPS_THROTTLE=1 SGLANG_DS_RPS_CLOSE=10 SGLANG_DS_RPS_OPEN=6 SGLANG_DS_RPS_WINDOW_S=2 \
$PY auto_benchmark_sglang.py --tight --timeline-gpu CUSTOM200 \
    --co --store-driven --real-backward --backward-subprocess --backward-mps-pct 10 \
    --port $PORT > /tmp/diag_rps_coserve.log 2>&1
cp output/server_tight_co.log output/server_200s_rpsdebug.log 2>/dev/null

echo "## RESTART daemon ##"
nvidia-cuda-mps-control -d 2>/dev/null; sleep 2
echo "daemon procs: $(ps aux | grep -c '[n]vidia-cuda-mps-control')"
echo "## DONE ##"
