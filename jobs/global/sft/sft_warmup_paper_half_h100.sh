#!/bin/bash
### ============================================================
### DTU HPC — Paper-style SFT warmup on 5k MOS examples, 1x H100
### Submit with: bsub < jobs/global/sft/sft_warmup_paper_half_h100.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-warmup-paper-half-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -o logs/sft_warmup_paper_half_h100_%J.out
#BSUB -e logs/sft_warmup_paper_half_h100_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh

# Hub repo to stream checkpoints into. Override at submit time with HUB_MODEL_ID=...
HUB_MODEL_ID="${HUB_MODEL_ID:-Leng2beat/speech-quality-assessement-qwen2audio-sft-warmup}"

torchrun --nproc_per_node=1 scripts/train/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --json-path data/processed/sft/train_nisqa_llama_10k.json \
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
    --wandb-run-name "sft-warmup-paper-half-h100" \
    --hub-model-id "$HUB_MODEL_ID" \
    --save-steps 200 \
    --save-total-limit 1

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
