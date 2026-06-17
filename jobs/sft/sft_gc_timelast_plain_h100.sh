#!/bin/bash
### ============================================================
### DTU HPC — PHASE-1 ARM A (basic twin): caption-last targets, vanilla loss,
### PLAIN free-text <|seconds|> timestamps, NO time mechanism.
###
### This is the fair basic-setup partner of sft_gc_timelast_timeaudio_h100.sh
### (ARM A, 28633522). The earlier basic-vs-TimeAudio comparison was run in the
### timestamp-FIRST order (sft_temporal_global_caption_plain_h100.sh vs
### sft_temporal_global_caption_timeaudio_h100.sh); ARM A then moved the
### timestamps to LAST but only on the TimeAudio mechanism, leaving the
### timestamp-last comparison one-sided. This job supplies the missing cell:
### basic setup, timestamps LAST, so basic vs TimeAudio is compared at the same
### answer order.
###
### Identical to sft_gc_timelast_timeaudio_h100.sh except the timestamp
### mechanism: same base model (Qwen2-Audio-7B), same caption-last data records
### (the <|s|> twin of ARM A's <a><f> set), same epochs (2), lr, batch,
### grad-accum, query prompt. The ONLY differences are:
###   - data: train_nisqa_temporal_gc_timelast_aug.json (plain <|s|>) instead of
###     the _anchoroffset (<a><f>) file
###   - NO --use-abs-time-embedding, NO --install-time-tokens (vanilla SFT)
### Keep the two in lockstep so the basic-vs-TimeAudio comparison stays clean.
###
### Save policy: FINAL ONLY, LOCAL ONLY (one ~16 GB model). Push to Hub
### manually afterward. Submit with:
###   bsub < jobs/sft/sft_gc_timelast_plain_h100.sh
###
### Memory audit (LSF rusage[mem] is PER-CORE): 48GB x 4 cores = 192 GB total,
### fits gpuh100 (~720 GB/node).
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-gc-timelast-plain-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -o logs/sft_gc_timelast_plain_h100_%J.out
#BSUB -e logs/sft_gc_timelast_plain_h100_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh

torchrun --nproc_per_node=1 scripts/train/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --json-path data/processed/temporal/train_nisqa_temporal_gc_timelast_aug.json \
    --data-root data \
    --model-name "$EXPERIMENT_DIR/models/sft_gc_timelast_plain_h100" \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 2 \
    --lr 1e-5 \
    --val-split 0 \
    --use-query-prompt \
    --wandb-run-name "sft-gc-timelast-plain-h100"

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
