#!/bin/sh
### ============================================================
### DTU HPC — SFT from base Qwen2-Audio: 5 epochs, cosine LR,
### warmup_ratio=0.03. Trains on the 9.5k filtered set (the 500-
### sample in-domain holdout test_nisqa_indomain.json is kept
### unseen by training).
###
### Replaces the +1/+2/+3 ladder with one continuous run.
### Submit with: bsub < jobs/sft/sft_base_5ep_cosine.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-base-5ep-cosine
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o logs/sft_base_5ep_cosine_%J.out
#BSUB -e logs/sft_base_5ep_cosine_%J.err

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

torchrun --nproc_per_node=1 src/asa/supervised-finetune.py \
    --model-name sft_base_5ep_cosine \
    --json-path data/processed/train_nisqa_llama_indomain.json \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 5 \
    --lr 1e-5 \
    --warmup-ratio 0.03 \
    --lr-scheduler-type cosine \
    --val-split 0.05 \
    --eval-steps 100 \
    --wandb-run-name "sft-base-5ep-cosine-warmup"

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
