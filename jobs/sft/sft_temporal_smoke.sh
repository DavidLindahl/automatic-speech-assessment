#!/bin/sh
### ============================================================
### DTU HPC — SFT Temporal smoke test from base Qwen2-Audio
### ============================================================

#BSUB -q gpul40s
#BSUB -J sft-temporal-smoke
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 12:00
#BSUB -o logs/sft_temporal_smoke_%J.out
#BSUB -e logs/sft_temporal_smoke_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
module load cuda/11.8
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

torchrun \
    --nproc_per_node=2 \
    scripts/train/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --model-name sft_temporal_smoke_qwen_base \
    --json-path data/processed/temporal/train_temporal_smoke.jsonl \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --epochs 1 \
    --eval-steps 50 \
    --wandb-project qwen2-audio-temporal-smoke \
    --wandb-run-name temporal-smoke-1k-baseqwen

echo "Training complete: $(date)"
