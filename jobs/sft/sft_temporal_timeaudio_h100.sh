#!/bin/bash
### ============================================================
### DTU HPC — TimeAudio temporal localization SFT, base Qwen2-Audio-7B, 1x H100
### Trains on anchor/offset <a><f> timestamp targets (timestamp-only, no
### category) with BOTH TimeAudio mechanisms on:
###   - mechanism 1: anchor/offset time tokens (--install-time-tokens)
###   - mechanism 2: learnable absolute-time frame embedding (--use-abs-time-embedding)
### Streams checkpoints to HF Hub (durable path). Submit with:
###   bsub < jobs/sft/sft_temporal_timeaudio_h100.sh
###
### Memory audit (LSF rusage[mem] is PER-CORE): 48GB x 4 cores = 192 GB total,
### fits gpuh100 (~720 GB/node). Matches the approved sft_full_paper_h100.sh.
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-temporal-timeaudio-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -o logs/sft_temporal_timeaudio_h100_%J.out
#BSUB -e logs/sft_temporal_timeaudio_h100_%J.err

source "$(dirname "$0")/../_lib/preamble.sh"

# Hub repo to stream checkpoints into. Override at submit time with HUB_MODEL_ID=...
HUB_MODEL_ID="${HUB_MODEL_ID:-Leng2beat/speech-quality-assessement-qwen2audio-temporal-timeaudio}"

torchrun --nproc_per_node=1 scripts/train/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --json-path data/processed/temporal/train_nisqa_temporal_anchoroffset.json \
    --data-root data \
    --model-name "$EXPERIMENT_DIR/models/sft_temporal_timeaudio_h100" \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 3 \
    --lr 1e-5 \
    --val-split 0 \
    --use-query-prompt \
    --use-abs-time-embedding \
    --install-time-tokens \
    --wandb-run-name "sft-temporal-timeaudio-h100" \
    --hub-model-id "$HUB_MODEL_ID" \
    --save-steps 200 \
    --save-total-limit 1

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
