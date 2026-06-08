#!/bin/bash
### ============================================================
### DTU HPC — SETUP 2: TimeAudio-method SFT, base Qwen2-Audio-7B, 1x H100.
### Trains on the new global-caption targets with anchor/offset <a><f>
### timestamps (global MOS caption + leading timestamp clause), with BOTH
### TimeAudio mechanisms on:
###   - mechanism 1: anchor/offset time tokens, numeral-seeded on BOTH the
###     input embedding and the (untied) lm_head (--install-time-tokens)
###   - mechanism 2: learnable, zero-init absolute-time frame embedding
###     (--use-abs-time-embedding)
###
### A/B partner of sft_temporal_global_caption_plain_h100.sh: identical base
### model, data size, epochs, lr, batch, grad-accum -- the ONLY difference is
### the timestamp mechanism (this one carries it). Both start from base
### Qwen2-Audio-7B so the comparison isolates the mechanism.
###
### NOTE: this needs the lm_head-seeding fix in src/asa/modeling_timeaudio.py.
### Confirm that file on DTU matches origin/feat/temporal-global-caption-dataset
### (grep write_seed) before submitting, or the seeding silently reverts to
### embed-only.
###
### Streams checkpoints to HF Hub (durable, quota-safe). Submit with:
###   bsub < jobs/sft/sft_temporal_global_caption_timeaudio_h100.sh
###
### Memory audit (LSF rusage[mem] is PER-CORE): 48GB x 4 cores = 192 GB total,
### fits gpuh100 (~720 GB/node).
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-temporal-gc-timeaudio-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -o logs/sft_temporal_gc_timeaudio_h100_%J.out
#BSUB -e logs/sft_temporal_gc_timeaudio_h100_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh

torchrun --nproc_per_node=1 scripts/train/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --json-path data/processed/temporal/train_nisqa_temporal_global_caption_aug_anchoroffset.json \
    --data-root data \
    --model-name "$EXPERIMENT_DIR/models/sft_temporal_gc_timeaudio_h100" \
    --hub-model-id Leng2beat/sft-temporal-global-caption-timeaudio \
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
    --wandb-run-name "sft-temporal-gc-timeaudio-h100"

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
