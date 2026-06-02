#!/bin/sh
### ============================================================
### DTU HPC — Evaluate the FULL 3-epoch timeaudio temporal model on the
### FOR temporal test set. Temporal IoU + category metrics, greedy.
### This is the clean full-run successor to the checkpoint-400 eval
### (28570755, epoch ~1.9, t-IoU 0.183). The model is the 16 GB final
### save from run 28571127 (both TimeAudio mechanisms on: anchor/offset
### time tokens + learnable absolute-time embedding). The checkpoint is a
### sharded safetensors save (4 shards), so the guard checks the index.
### Submit with: bsub < jobs/evaluate/evaluate_timeaudio_full_temporal.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-timeaudio-full
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 4:00
#BSUB -o logs/eval_timeaudio_full_%J.out
#BSUB -e logs/eval_timeaudio_full_%J.err

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

MODEL_PATH="$PROJECT_DIR/models/sft_temporal_timeaudio_h100"
DATASET_PATH="data/processed/temporal/test_FOR_temporal.json"
OUTPUT_DIR="results/evaluation/temporal/timeaudio_full_FOR"

echo "=========================================="
echo "Job ID   : ${LSB_JOBID:-local}"
echo "Host     : $(hostname)"
echo "Model    : $MODEL_PATH (full 3-epoch run 28571127)"
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
echo "Temporal eval (full 3-epoch) complete: $(date)"
echo "=========================================="
