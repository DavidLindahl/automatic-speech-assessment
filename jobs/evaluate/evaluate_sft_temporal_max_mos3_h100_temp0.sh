#!/bin/sh
### ============================================================
### DTU HPC — Evaluate temporal max_mos3 checkpoint on H100, temp=0
### Submit with: bsub < jobs/evaluate/evaluate_sft_temporal_max_mos3_h100_temp0.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-sft-temporal-max-mos3-h100-temp0
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 23:59
#BSUB -o logs/evaluate_temporal_max_mos3_h100_temp0_%J.out
#BSUB -e logs/evaluate_temporal_max_mos3_h100_temp0_%J.err

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

DEFAULT_HUB_MODEL="Leng2beat/speech-quality-assessement-qwen2audio-sft-temporal-max-mos3-partial-step305"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_HUB_MODEL}"
DATASET_PATH="data/processed/train_nisqa_temporal_mix_max_mos3.json"
OUTPUT_DIR="${OUTPUT_DIR:-results/evaluation/sft_temporal_max_mos3_h100_temp0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"

if [ -d "$MODEL_PATH" ]; then
  if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/pytorch_model.bin" ]; then
    echo "Local model dir exists but no model weights found in: $MODEL_PATH"
    echo "Falling back to Hub model: $DEFAULT_HUB_MODEL"
    MODEL_PATH="$DEFAULT_HUB_MODEL"
  fi
fi

echo "=========================================="
echo "Job ID      : ${LSB_JOBID:-local}"
echo "Host        : $(hostname)"
echo "GPUs        : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Model       : $MODEL_PATH"
echo "Dataset     : $DATASET_PATH"
echo "Output      : $OUTPUT_DIR"
echo "Batch       : $BATCH_SIZE"
echo "Temperature : $TEMPERATURE"
echo "Decode      : greedy"
echo "HF Cache    : $HUGGINGFACE_HUB_CACHE"
echo "Started     : $(date)"
echo "=========================================="

nvidia-smi

if [ ! -f "$DATASET_PATH" ]; then
  echo "Missing dataset file: $DATASET_PATH"
  echo "Run first: bsub < jobs/train/build_nisqa_temporal_max_json.sh"
  exit 1
fi

uv run python scripts/eval/evaluate_temporal.py \
  --model-path "$MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --data-root data \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --greedy \
  --temperature "$TEMPERATURE" \
  --use-query-prompt

echo "=========================================="
echo "Temporal evaluation complete: $(date)"
echo "=========================================="
