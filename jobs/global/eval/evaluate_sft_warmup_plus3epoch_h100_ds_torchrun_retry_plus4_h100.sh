#!/bin/sh
### ============================================================
### DTU HPC LSF job script — Model Evaluation
### ============================================================
#BSUB -q gpuh100
#BSUB -J eval-sft-warmup-plus3epoch-h100-ds-torchrun-retry-plus4-h100
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
mkdir -p results/inference/sft

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

MODEL_PATH="models/sft_warmup_plus3epoch_h100_ds_torchrun_retry_plus4"
OUTPUT_PATH="results/evaluation/sft/sft_warmup_plus3epoch_h100_ds_torchrun_retry_plus4_eval_h100"
DATASETS=(
    "data/processed/eval/test_FOR.json"
    "data/processed/eval/test_LIVE.json"
    "data/processed/eval/test_P501.json"
)

echo "Evaluating datasets: ${DATASETS[*]}"
uv run python scripts/eval/evaluate.py     --model-path "$MODEL_PATH"     --output-dir "$OUTPUT_PATH"     --dataset-path "${DATASETS[0]}"     --dataset-path "${DATASETS[1]}"     --dataset-path "${DATASETS[2]}"     --batch-size 8

echo ""
echo "=========================================="
echo "Evaluation complete: $(date)"
echo "=========================================="
