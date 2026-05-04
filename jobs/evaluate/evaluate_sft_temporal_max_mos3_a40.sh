#!/bin/sh
### ============================================================
### DTU HPC — Evaluate temporal checkpoint from Hugging Face Hub
### Submit with: bsub < jobs/evaluate/evaluate_sft_temporal_hub_step305_a40.sh
### ============================================================

#BSUB -q gpua40
#BSUB -J eval-sft-temporal-hub-step305-a40
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 12:00
#BSUB -o logs/evaluate_temporal_hub_step305_%J.out
#BSUB -e logs/evaluate_temporal_hub_step305_%J.err

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

MODEL_PATH="Leng2beat/speech-quality-assessement-qwen2audio-sft-temporal-max-mos3-partial-step305"
DATASET_PATH="data/processed/train_nisqa_temporal_mix_max_mos3.json"
OUTPUT_DIR="results/evaluation/sft_temporal_hub_step305"

echo "=========================================="
echo "Job ID   : ${LSB_JOBID:-local}"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Model    : $MODEL_PATH"
echo "Dataset  : $DATASET_PATH"
echo "Output   : $OUTPUT_DIR"
echo "HF Cache : $HUGGINGFACE_HUB_CACHE"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi

if [ ! -f "$DATASET_PATH" ]; then
  echo "Missing dataset file: $DATASET_PATH"
  echo "Run first: bsub < jobs/train/build_nisqa_temporal_max_json.sh"
  exit 1
fi

uv run python src/asa/evaluate_temporal.py \
  --model-path "$MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --data-root data \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 4 \
  --greedy \
  --use-query-prompt

echo "=========================================="
echo "Temporal Hub evaluation complete: $(date)"
echo "=========================================="
