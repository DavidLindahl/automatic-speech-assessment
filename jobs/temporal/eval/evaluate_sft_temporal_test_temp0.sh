#!/bin/bash
### ============================================================
### DTU HPC — Evaluate local temporal SFT on temporal test sets
### Submit with: bsub < jobs/evaluate/evaluate_sft_temporal_test_temp0.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-sft-temporal-test-temp0
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 4:00
#BSUB -o logs/evaluate_sft_temporal_test_temp0_%J.out
#BSUB -e logs/evaluate_sft_temporal_test_temp0_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/results/evaluation/temporal"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

HF_CACHE_ROOT="/tmp/${USER:-s234817}/hf_cache"
export HF_HOME="${HF_HOME:-$HF_CACHE_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"

MODEL_PATH="${MODEL_PATH:-$EXPERIMENT_DIR/models/sft_temporal}"
OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_DIR/results/evaluation/temporal/sft_temporal_test_temp0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"

DATASETS=(
  "data/processed/temporal/test_FOR_temporal.json"
  "data/processed/temporal/test_LIVE_temporal.json"
  "data/processed/temporal/test_P501_temporal.json"
)

echo "=========================================="
echo "Job ID      : ${LSB_JOBID:-local}"
echo "Host        : $(hostname)"
echo "GPUs        : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Model       : $MODEL_PATH"
echo "Output      : $OUTPUT_DIR"
echo "Batch       : $BATCH_SIZE"
echo "Temperature : $TEMPERATURE"
echo "Decode      : greedy"
echo "HF_HOME     : $HF_HOME"
echo "HF hub cache: $HUGGINGFACE_HUB_CACHE"
echo "Started     : $(date)"
echo "=========================================="

nvidia-smi

if [ ! -d "$MODEL_PATH" ]; then
  echo "Missing model directory: $MODEL_PATH"
  exit 1
fi

for dataset_path in "${DATASETS[@]}"; do
  if [ ! -f "$dataset_path" ]; then
    echo "Missing dataset file: $dataset_path"
    exit 1
  fi
done

uv run python scripts/eval/evaluate_temporal.py \
  --model-path "$MODEL_PATH" \
  --dataset-path "${DATASETS[0]}" \
  --dataset-path "${DATASETS[1]}" \
  --dataset-path "${DATASETS[2]}" \
  --data-root data \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --greedy \
  --temperature "$TEMPERATURE" \
  --use-query-prompt

echo "=========================================="
echo "Temporal test evaluation complete: $(date)"
echo "=========================================="
