#!/usr/bin/env bash
# Orchestrate the MPS de-risk probe across 3 conditions on GPU 0.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="python $HERE/mps_probe.py"
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log

stop_mps() { echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 1; pkill -f nvidia-cuda-mps 2>/dev/null; sleep 1; }
start_mps() { mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; nvidia-cuda-mps-control -d; sleep 2; }

echo "### cleanup any prior MPS"; stop_mps

echo; echo "### A: victim alone (baseline)"
$PY --role victim --iters 60 --sleep-ms 50 2>&1 | grep -E "VICTIM_RESULT"

echo; echo "### B: victim + hog, NO MPS (default time-slice)"
$PY --role hog --seconds 14 >/tmp/mps_hogB.log 2>&1 &
HPID=$!
sleep 2   # let hog warm up and start firing
$PY --role victim --iters 60 --sleep-ms 50 2>&1 | grep -E "VICTIM_RESULT"
wait $HPID 2>/dev/null; grep HOG_RESULT /tmp/mps_hogB.log

echo; echo "### C: victim + hog, MPS daemon, hog capped @ 10%"
start_mps
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10 $PY --role hog --seconds 14 >/tmp/mps_hogC.log 2>&1 &
HPID=$!
sleep 2
# victim runs at default (uncapped) MPS share
$PY --role victim --iters 60 --sleep-ms 50 2>&1 | grep -E "VICTIM_RESULT"
wait $HPID 2>/dev/null; grep HOG_RESULT /tmp/mps_hogC.log
stop_mps
echo; echo "### DONE"
