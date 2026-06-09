#!/usr/bin/env bash
# reproduce_compare_figure.sh — one-shot repro of plots/compare_slora_dserve_*.png
# (the SLoRA-vs-DeltaServe co-serving figure: inference E2E latency for both runs
#  + DeltaServe FT-throughput band that fills the troughs between request bursts).
#
# It runs the SAME 4-step pipeline used to make the figure:
#   1) inference-only baseline   (auto_benchmark_sglang.py, no --co)   -> SLoRA line
#   2) co-serving run            (auto_benchmark_sglang.py  --co ...)  -> DeltaServe line + bwd fires
#   3) adapt both outputs        (make_compare_inputs.py)
#   4) plot                      (compare_slora_dserve.py)             -> the PNG
#
# Run it from sglang-port/v046-port/ :   bash reproduce_compare_figure.sh
# Override anything via env vars, e.g.:
#   MODEL=/path/to/Meta-Llama-3-8B SHAPE=tight GPU_NAME="H200" bash reproduce_compare_figure.sh
#
# Requirements that must already be set up (all tracked in the repo EXCEPT the
# absolute paths inside the config, which you adjust for your box):
#   - the timeline:   eval/llama3/timelines/<auto-detected gpu>/timeline_<SHAPE>.csv
#   - the FT config:  configs/serving_config_finetuning_llama3_both.yaml
#       └─ edit its finetuning_lora_path / data_path / model_path to match THIS box
#   - nvidia-cuda-mps-control -d running (MPS daemon) for the backward cap
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python}"
MODEL="${MODEL:-/mnt/weka/home/jianshu.she/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B}"
SHAPE="${SHAPE:-tight}"                 # picks eval/llama3/timelines/<gpu>/timeline_<SHAPE>.csv
GPU_NAME="${GPU_NAME:-H200}"            # figure title only
CONFIG="${CONFIG:-configs/serving_config_finetuning_llama3_both.yaml}"
PORT="${PORT:-30200}"
MPS_PCT="${MPS_PCT:-10}"               # backward MPS cap (co-serving)
OUT=output
mkdir -p "$OUT" plots

echo "### [1/4] inference-only baseline (SLoRA line) ###"
$PY auto_benchmark_sglang.py --$SHAPE --no-radix \
    --model "$MODEL" --port "$PORT"

echo "### [2/4] co-serving run (DeltaServe line + FT throughput) ###"
$PY auto_benchmark_sglang.py --$SHAPE --co --store-driven --real-backward \
    --model "$MODEL" --port "$PORT" \
    --finetune-config "$CONFIG" --backward-mps-pct "$MPS_PCT"

INF_CSV="$OUT/timeline_results_${SHAPE}_inf.csv"
CO_CSV="$OUT/timeline_results_${SHAPE}_co_real_sub.csv"
CO_LOG="$OUT/server_${SHAPE}_co.log"

echo "### [3/4] adapt outputs -> plotter inputs ###"
$PY make_compare_inputs.py --results-in "$INF_CSV" --results-out "$OUT/slora.csv"
$PY make_compare_inputs.py --results-in "$CO_CSV"  --results-out "$OUT/dserve.csv" \
    --server-log "$CO_LOG" --bwd-out "$OUT/bwd.csv"

echo "### [4/4] plot ###"
# auto-detect the gpu timeline subdir the same way auto_benchmark does
GPU_SUB=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | grep -qi A100 && echo A100 || echo 5090)
TL="../../eval/llama3/timelines/${GPU_SUB}/timeline_${SHAPE}.csv"
$PY compare_slora_dserve.py \
    --slora "$OUT/slora.csv" --dserve "$OUT/dserve.csv" --bwd "$OUT/bwd.csv" \
    --timeline "$TL" --gpu-name "$GPU_NAME" \
    --output "plots/compare_slora_dserve_${SHAPE}.png"

echo "### done -> plots/compare_slora_dserve_${SHAPE}.png ###"
