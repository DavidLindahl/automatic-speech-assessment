#!/bin/sh
### ============================================================
### DTU HPC — SFT AB Test (Medium Test: 5000 samples, ~1 hour)
### Submit with: bsub < jobs/sft/sft_ab_warmup.sh
### ============================================================

#BSUB -q gpua100
#BSUB -J sft-ab-warmup-a100
#BSUB -n 8
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 10:00
#BSUB -o logs/sft_ab_warmup_%J.out
#BSUB -e logs/sft_ab_warmup_%J.err

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
# If OOM occurs, try reducing --batch-size or increasing ds_zero offloading.
torchrun \
    --nproc_per_node=2 \
    src/asa/supervised-finetune-ab.py \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --max-samples 5000 \
    --batch-size 4 \
    --epochs 2 \
    --eval-steps 50 \
    --model-name "sft_ab_warmup" \
    --wandb-run-name "sft-ab-warmup"

echo "Training complete: $(date)"
