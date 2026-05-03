#!/bin/sh
### ============================================================
### DTU HPC — Evaluate temporal max_mos3 checkpoint
### Submit with: bsub < jobs/evaluate/evaluate_sft_temporal_max_mos3_a40.sh
### ============================================================

#BSUB -q gpua40
#BSUB -J eval-sft-temporal-max-mos3-a40
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 12:00
#BSUB -o logs/evaluate_temporal_max_mos3_%J.out
#BSUB -e logs/evaluate_temporal_max_mos3_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONPATH=src

MODEL_PATH="models/sft_temporal_max_mos3"
DATASET_PATH="data/processed/train_nisqa_temporal_mix_max_mos3.json"
OUTPUT_DIR="results/evaluation/sft_temporal_max_mos3"

echo "=========================================="
echo "Job ID   : ${LSB_JOBID:-local}"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Model    : $MODEL_PATH"
echo "Dataset  : $DATASET_PATH"
echo "Output   : $OUTPUT_DIR"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi

if [ ! -f "$DATASET_PATH" ]; then
  echo "Missing dataset file: $DATASET_PATH"
  echo "Run first: bsub < jobs/train/build_nisqa_temporal_max_json.sh"
  exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
  echo "Missing model checkpoint directory: $MODEL_PATH"
  echo "Run first: bsub < jobs/sft/sft_temporal_max_mos3.sh"
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
echo "Temporal evaluation complete: $(date)"
echo "=========================================="
