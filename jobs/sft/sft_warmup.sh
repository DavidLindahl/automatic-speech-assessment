#!/bin/sh
### ============================================================
### DTU HPC — SFT Medium Test (500 samples, ~1 hour)
### Submit with: bsub < jobs/sft/sft_warmup.sh
### ============================================================

#BSUB -q gpul40s
#BSUB -J sft-medium
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 3:00
#BSUB -o logs/sft_medium_%J.out
#BSUB -e logs/sft_medium_%J.err

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
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi



torchrun \
    --nproc_per_node=2 \
    src/asa/supervised-finetune.py \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --max-samples 5000 \
    --batch-size 4 \
    --epochs 2 \
    --eval-steps 50 \
    --wandb-run-name "sft-warmup"

echo "Training complete: $(date)"
