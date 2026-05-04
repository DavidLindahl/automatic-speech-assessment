#!/bin/sh
### ============================================================
### DTU HPC — SFT continuation from +2 epoch checkpoint on 1x H100
### Retry with DeepSpeed + torchrun single-process launch
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-warmup-plus3-h100-ds-tr-plus3
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 12:00
#BSUB -o logs/sft_warmup_plus3epoch_h100_ds_torchrun_retry_plus3_%J.out
#BSUB -e logs/sft_warmup_plus3epoch_h100_ds_torchrun_retry_plus3_%J.err

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
    --nproc_per_node=1 \
    src/asa/supervised-finetune.py \
    --model-id models/sft_warmup_plus2epoch_l40s \
    --model-name sft_warmup_plus3epoch_h100_ds_torchrun_retry_plus3 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 2 \
    --epochs 1 \
    --gradient-accumulation-steps 4 \
    --val-split 0.05 \
    --eval-steps 50 \
    --wandb-run-name "sft-warmup-plus3epoch-h100-ds-torchrun-retry-plus3"

echo "Training complete: $(date)"
