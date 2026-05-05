#!/bin/bash
### ============================================================
### DTU HPC — Paper-style SFT warmup on 5k MOS examples, 1x H100
### Submit with: bsub < jobs/sft/sft_warmup_paper_half_h100.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-warmup-paper-half-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=88GB]"
#BSUB -M 88GB
#BSUB -W 24:00
#BSUB -o logs/sft_warmup_paper_half_h100_%J.out
#BSUB -e logs/sft_warmup_paper_half_h100_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/models"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export HF_HOME="$EXPERIMENT_DIR/.cache/huggingface"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

torchrun --nproc_per_node=1 src/asa/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --json-path data/processed/train_nisqa_llama_10k.json \
    --data-root data \
    --model-name "$EXPERIMENT_DIR/models/sft_warmup_paper_half_h100" \
    --max-samples 5000 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 2 \
    --lr 1e-5 \
    --val-split 0 \
    --wandb-run-name "sft-warmup-paper-half-h100"

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
