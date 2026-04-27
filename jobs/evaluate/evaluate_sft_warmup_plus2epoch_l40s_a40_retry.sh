#!/bin/sh
### ============================================================
### DTU HPC LSF job script — Model Evaluation
### ============================================================
#BSUB -q gpua40
#BSUB -J eval-sft-warmup-plus2epoch-l40s-a40-retry
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

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi

MODEL_PATH="models/sft_warmup_plus1epoch"
OUTPUT_PATH="results/evaluation/sft_warmup_plus1epoch_eval_a40"
DATASETS=(
    "data/processed/test_FOR.json"
    "data/processed/test_LIVE.json"
    "data/processed/test_P501.json"
)

echo "Evaluating datasets: ${DATASETS[*]}"
uv run python src/asa/evaluate.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_PATH" \
    --dataset-path "${DATASETS[0]}" \
    --dataset-path "${DATASETS[1]}" \
    --dataset-path "${DATASETS[2]}" \
    --batch-size 8

echo ""
echo "=========================================="
echo "Evaluation complete: $(date)"
echo "=========================================="
