#!/bin/sh
### ============================================================
### DTU HPC — Evaluate temporal smoke-test checkpoint
### ============================================================

#BSUB -q gpua40
#BSUB -J eval-sft-temporal-smoke-a40
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 4:00
#BSUB -o logs/evaluate_temporal_smoke_%J.out
#BSUB -e logs/evaluate_temporal_smoke_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
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

uv run python src/asa/evaluate_temporal.py \
    --model-path models/sft_temporal_smoke_qwen_base \
    --dataset-path data/processed/test_temporal_smoke.jsonl \
    --output-dir results/evaluation/sft_temporal_smoke_qwen_base \
    --batch-size 4

echo ""
echo "=========================================="
echo "Temporal evaluation complete: $(date)"
echo "=========================================="
