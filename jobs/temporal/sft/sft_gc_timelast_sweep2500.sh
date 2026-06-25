#!/bin/bash
### ============================================================
### DTU HPC — PHASE-1 ARM A: caption-last targets, vanilla loss (order ablation)
### Trains the TimeAudio-mechanism model on the NEW caption-first /
### timestamps-LAST targets (global-caption-timelast-anchoroffset), with the
### stock cross-entropy loss. Identical to 28615749 except the target order:
### same base model, same data records, same epochs (2), lr, batch, grad-accum,
### both TimeAudio mechanisms on. Isolates the answer-order effect (T1).
###
### A/B/C family:
###   28615749 (timestamp-first, vanilla CE)  -> this job (timestamp-LAST,
###   vanilla CE) -> sft_gc_timelast_softloss_h100.sh (timestamp-LAST,
###   weighted + distance-aware CE).
###
### Save policy: FINAL ONLY, LOCAL ONLY (one ~16 GB model). Push to Hub
### manually afterward. Submit with:
###   bsub < jobs/sft/sft_gc_timelast_timeaudio_h100.sh
###
### Memory audit (LSF rusage[mem] is PER-CORE): 48GB x 4 cores = 192 GB total,
### fits gpuh100 (~720 GB/node).
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-gc-sweep2500
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -o logs/sft_gc_sweep2500_%J.out
#BSUB -e logs/sft_gc_sweep2500_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh

torchrun --nproc_per_node=1 scripts/train/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --json-path data/processed/temporal/train_gc_timelast_sweep_2500.json \
    --data-root data \
    --model-name "$EXPERIMENT_DIR/models/sft_gc_timelast_sweep2500" \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 2 \
    --lr 1e-5 \
    --val-split 0 \
    --use-query-prompt \
    --use-abs-time-embedding \
    --install-time-tokens \
    --wandb-run-name "sft-gc-timelast-sweep2500"

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
