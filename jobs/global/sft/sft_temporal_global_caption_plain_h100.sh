#!/bin/bash
### ============================================================
### DTU HPC — SETUP 1: plain SFT, standard <|seconds|> timestamps, base
### Qwen2-Audio-7B, 1x H100. Trains on the new global-caption targets
### (global MOS caption + leading timestamp clause). NO time tokens, NO
### absolute-time embedding -- vanilla SFT.
###
### This is the A/B partner of sft_temporal_global_caption_timeaudio_h100.sh:
### identical base model, data size, epochs, lr, batch, grad-accum. The ONLY
### difference is the timestamp mechanism (this one: none; the other: TimeAudio
### tokens + embedding). Keep the two in lockstep so the comparison is clean.
###
### Streams checkpoints to HF Hub (durable, quota-safe). Submit with:
###   bsub < jobs/sft/sft_temporal_global_caption_plain_h100.sh
###
### Memory audit (LSF rusage[mem] is PER-CORE): 48GB x 4 cores = 192 GB total,
### fits gpuh100 (~720 GB/node).
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-temporal-gc-plain-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -o logs/sft_temporal_gc_plain_h100_%J.out
#BSUB -e logs/sft_temporal_gc_plain_h100_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh

torchrun --nproc_per_node=1 scripts/train/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --json-path data/processed/temporal/train_nisqa_temporal_global_caption_aug.json \
    --data-root data \
    --model-name "$EXPERIMENT_DIR/models/sft_temporal_gc_plain_h100" \
    --hub-model-id Leng2beat/sft-temporal-global-caption-plain \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 3 \
    --lr 1e-5 \
    --val-split 0 \
    --use-query-prompt \
    --wandb-run-name "sft-temporal-gc-plain-h100"

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
