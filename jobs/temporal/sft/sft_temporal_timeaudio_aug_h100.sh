#!/bin/bash
### ============================================================
### DTU HPC — TimeAudio temporal localization SFT on the PLACEMENT-AUGMENTED set,
### base Qwen2-Audio-7B, 1x H100. Trains on anchor/offset <a><f> timestamp targets
### (timestamp-only, no category) with BOTH TimeAudio mechanisms on:
###   - mechanism 1: anchor/offset time tokens (--install-time-tokens)
###   - mechanism 2: learnable absolute-time frame embedding (--use-abs-time-embedding)
###
### Only the training data differs from sft_temporal_timeaudio_h100.sh: this trains
### on the augmented JSONL (13,495 records, ~2.63x via non-overlapping REF reuse,
### data-gen job 28603841) instead of the 5,136-record single-placement baseline.
### The augmented vs baseline IoU delta is the experiment.
###
### Submit with: bsub < jobs/temporal/sft/sft_temporal_timeaudio_aug_h100.sh
###
### Memory audit (LSF rusage[mem] is PER-CORE): 48GB x 4 cores = 192 GB total,
### fits gpuh100 (~720 GB/node). Identical to the approved baseline.
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-temporal-timeaudio-aug-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -o logs/sft_temporal_timeaudio_aug_h100_%J.out
#BSUB -e logs/sft_temporal_timeaudio_aug_h100_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh

# Final-only local save (no Hub streaming). Without --hub-model-id the trainer
# sets save_strategy="no" + save_only_model=True, so exactly ONE ~16 GB
# checkpoint is written at the end. No mid-run checkpoints, no quota spike.
torchrun --nproc_per_node=1 scripts/train/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --json-path data/processed/temporal/train_nisqa_temporal_anchoroffset_aug.json \
    --data-root data \
    --model-name "$EXPERIMENT_DIR/models/sft_temporal_timeaudio_aug_h100" \
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
    --wandb-run-name "sft-temporal-timeaudio-aug-h100"

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
