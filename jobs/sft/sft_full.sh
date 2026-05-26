#!/bin/sh
### ============================================================
### DTU HPC — SFT Full Training (10k samples, 2 epochs)
### Submit with: bsub < jobs/sft/sft_full.sh
### ============================================================

#BSUB -q gpul40s
#BSUB -J sft-full
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 24:00
#BSUB -o logs/sft_full_%J.out
#BSUB -e logs/sft_full_%J.err

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
    --model-name sft_full \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --epochs 2 \
    --eval-steps 100 \
    --wandb-run-name "full-10k-2ep"

echo "Training complete: $(date)"
