#!/bin/sh
### ============================================================
### DTU HPC — Evaluate timeaudio checkpoint-400 (epoch ~1.9) on the
### FOR temporal test set. Temporal IoU + category metrics, greedy.
### checkpoint-400 is the intact partial checkpoint from the
### disk-full-killed run 28564567 (the final top-level save was
### corrupted). Paths are hardcoded (the env-var override on the
### generic script did not survive bsub, and its existence guard
### rejects sharded safetensors).
### Submit with: bsub < jobs/evaluate/evaluate_timeaudio_ckpt400_temporal.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-timeaudio-ckpt400
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 4:00
#BSUB -o logs/eval_timeaudio_ckpt400_%J.out
#BSUB -e logs/eval_timeaudio_ckpt400_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONPATH=src
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HUGGINGFACE_HUB_CACHE"

MODEL_PATH="$PROJECT_DIR/models/sft_temporal_timeaudio_h100/checkpoint-400"
DATASET_PATH="data/processed/temporal/test_FOR_temporal.json"
OUTPUT_DIR="results/evaluation/temporal/timeaudio_ckpt400_FOR"

echo "=========================================="
echo "Job ID   : ${LSB_JOBID:-local}"
echo "Host     : $(hostname)"
echo "Model    : $MODEL_PATH"
echo "Dataset  : $DATASET_PATH"
echo "Output   : $OUTPUT_DIR"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
  echo "ERROR: no sharded checkpoint index at $MODEL_PATH"
  exit 1
fi
if [ ! -f "$DATASET_PATH" ]; then
  echo "ERROR: missing dataset $DATASET_PATH"
  exit 1
fi

uv run python scripts/eval/evaluate_temporal.py \
  --model-path "$MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --data-root data \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 4 \
  --greedy \
  --temperature 0.0 \
  --use-query-prompt

echo "=========================================="
echo "Temporal eval (ckpt400) complete: $(date)"
echo "=========================================="
