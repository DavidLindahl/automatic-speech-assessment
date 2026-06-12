#!/bin/bash
### ============================================================
### DTU HPC — PHASE-1 ARM B: caption-last targets + temporal loss (the fix)
### Trains the TimeAudio-mechanism model on the caption-first /
### timestamps-LAST targets WITH the Phase-1 temporal loss:
###   - time-token CE weight lambda = 5  (timestamps carry ~8% of tokens;
###     weighting shifts loss mass onto the task tokens)
###   - distance-aware Gaussian soft targets sigma = 1 bucket over the
###     ordered anchor (1 s) and offset (0.1 s) vocabularies, so near misses
###     get partial credit and the gradient has a direction along time
###
### Identical to sft_gc_timelast_timeaudio_h100.sh except the two loss flags;
### that job is the loss ablation partner. Motivated by frame probe 28627256:
### the degradation location is linearly readable from the model's own
### features (probe t-IoU 0.52-0.55) while the CE-trained readout collapsed
### to the interval prior (0.10). Target band for this run: t-IoU > 0.36
### (beats every audio-blind strategy), ideally toward the probe's ~0.5.
###
### Save policy: FINAL ONLY, LOCAL ONLY (one ~16 GB model). Push to Hub
### manually afterward. Submit with:
###   bsub < jobs/sft/sft_gc_timelast_softloss_h100.sh
###
### Memory audit (LSF rusage[mem] is PER-CORE): 48GB x 4 cores = 192 GB total,
### fits gpuh100 (~720 GB/node).
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-gc-timelast-softloss-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -o logs/sft_gc_timelast_softloss_h100_%J.out
#BSUB -e logs/sft_gc_timelast_softloss_h100_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh

torchrun --nproc_per_node=1 scripts/train/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --json-path data/processed/temporal/train_nisqa_temporal_gc_timelast_aug_anchoroffset.json \
    --data-root data \
    --model-name "$EXPERIMENT_DIR/models/sft_gc_timelast_softloss_h100" \
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
    --time-token-loss-weight 5.0 \
    --time-token-soft-sigma 1.0 \
    --wandb-run-name "sft-gc-timelast-softloss-h100"

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
