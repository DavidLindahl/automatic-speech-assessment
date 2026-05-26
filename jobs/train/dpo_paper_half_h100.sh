#!/bin/bash
### ============================================================
### DTU HPC — DPO/ALLD from paper-style half-data warmup, 1x H100
### Submit with: bsub < jobs/train/dpo_paper_half_h100.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-paper-half-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_paper_half_h100_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_paper_half_h100_%J.err

source "$(dirname "$0")/../_lib/preamble.sh"

# Hub repo to stream checkpoints into. Override at submit time with HUB_MODEL_ID=...
HUB_MODEL_ID="${HUB_MODEL_ID:-Leng2beat/speech-quality-assessement-qwen2audio-dpo-paper-half}"

torchrun --nproc_per_node=1 src/asa/dpo-finetune.py \
    --model-name "$EXPERIMENT_DIR/models/dpo_paper_half_h100" \
    --model-id "$EXPERIMENT_DIR/models/sft_warmup_paper_half_h100" \
    --json-path "$EXPERIMENT_DIR/data/processed/train_dpo_paper_half_h100_clean.json" \
    --data-root data \
    --batch-size 2 \
    --epochs 1 \
    --lr 5e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --wandb-run-name "dpo-paper-half-h100" \
    --hub-model-id "$HUB_MODEL_ID" \
    --save-steps 200 \
    --save-total-limit 1

echo "=========================================="
echo "DPO training complete: $(date)"
echo "=========================================="
