#!/bin/sh
### ============================================================
### DTU HPC LSF job script — DPO Training Qwen2-Audio (+2epoch L40S)
### Submit with: bsub < jobs/global/alld/dpo_plus2epoch_l40s_h100.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-plus2epoch-l40s-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=88GB]"
#BSUB -M 88GB
#BSUB -W 24:00
#BSUB -o logs/dpo_plus2epoch_l40s_h100_%J.out
#BSUB -e logs/dpo_plus2epoch_l40s_h100_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p models/dpo_plus2epoch_l40s_h100

module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi

torchrun \
    --nproc_per_node=1 \
    scripts/train/dpo-finetune.py \
    --model-name dpo_plus2epoch_l40s_h100 \
    --model-id models/sft_warmup_plus2epoch_l40s \
    --json-path "data/processed/dpo/train_dpo_10k_plus2epoch_l40s.json" \
    --data-root "data" \
    --batch-size 2 \
    --epochs 1 \
    --lr 1e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed "configs/ds_zero2.json" \
    --wandb-run-name "dpo-plus2epoch-l40s-h100"

echo "=========================================="
echo "DPO Training complete: $(date)"
echo "=========================================="
