#!/bin/sh
### ============================================================
### DTU HPC — SFT AB Full Training (10k samples, 2 epochs)
### Submit with: bsub < jobs/sft/sft_ab_full.sh
### ============================================================

#BSUB -q gpul40s
#BSUB -J sft-ab-full
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o logs/sft_ab_full_%J.out
#BSUB -e logs/sft_ab_full_%J.err

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

# Note: AB test loads 2 audios per sample.
# If OOM occurs, try reducing --batch-size to 2.
torchrun \
    --nproc_per_node=2 \
    src/asa/supervised-finetune-ab.py \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --epochs 2 \
    --eval-steps 100 \
    --model-name "sft_ab_full" \
    --wandb-run-name "full-ab-10k-2ep"

echo "Training complete: $(date)"
