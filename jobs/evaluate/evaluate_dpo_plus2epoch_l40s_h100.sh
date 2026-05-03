#!/bin/sh
### ============================================================
### DTU HPC LSF job script — DPO Model Evaluation (H100)
### Model: models/dpo_plus2epoch_l40s_h100
### ============================================================
#BSUB -q gpuh100
#BSUB -J eval-dpo-plus2epoch-l40s-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 4:00
#BSUB -o logs/evaluate_%J.out
#BSUB -e logs/evaluate_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p results/inference/dpo

module load cuda/11.8 || true
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi

MODEL_PATH="models/dpo_plus2epoch_l40s_h100"
OUTPUT_PATH="results/evaluation/dpo_plus2epoch_l40s_h100_eval_h100"
DATASETS=(
    "data/processed/test_FOR.json"
    "data/processed/test_LIVE.json"
    "data/processed/test_P501.json"
)

echo "Evaluating datasets: ${DATASETS[*]}"
uv run python src/asa/evaluate.py     --model-path "$MODEL_PATH"     --output-dir "$OUTPUT_PATH"     --dataset-path "${DATASETS[0]}"     --dataset-path "${DATASETS[1]}"     --dataset-path "${DATASETS[2]}"     --batch-size 8

echo ""
echo "=========================================="
echo "Evaluation complete: $(date)"
echo "=========================================="
