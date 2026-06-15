#!/bin/bash
### ============================================================
### DTU HPC — MEDIUM-difficulty timestamp-cycle DPO on ARM A (Goldilocks shift).
### Submit with: bsub < jobs/temporal/alld/dpo_temporal_armA_medshift.sh
###
### The negative-difficulty experiment. Diagnosis (W&B + data): on the IoU-0.88
### caption-last SFT base, temporal ALLD never beats SFT because the preference
### signal is ~0, the model's own sampled intervals land within 0.4 s of gold on
### 77% of clips, so the reward margin stays flat at 0. The graded-jitter set
### (0.5-4 s) went too far and COLLAPSED the model. This arm is the middle: a
### single FIXED 0.75 s interval shift on the gold interval (data
### train_dpo_armA_medshift.json, built by build_dpo_medium_shift.py), big enough
### to be a learnable signal but small enough to stay near the true window.
### Single-factor: caption + MOS gold, only the interval clause differs.
###
### Tests whether ANY negative difficulty lets ALLD beat SFT (0.884/0.896/0.871),
### or whether the model collapses the moment the signal becomes learnable.
###
###   - policy = ARM A (sft_gc_timelast_timeaudio_h100). TimeAudio subclass
###     auto-selected so abs_time_embedding survives + keeps training.
###   - --dims-source-json: REQUIRED for the temporal reference prompt.
###   - AUDIO fix 51380b9 in effect.
###
### Same hypers as every cycle: LR 1e-6, beta 0.4, 1 epoch, batch 2, grad-accum 16.
### --final-save-only to the public Hub. Mem: -n 4 x rusage[mem=64GB] = 256 GB,
### fits gpuh100 (~720 GB).
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-temporal-armA-medshift
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_armA_medshift_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_armA_medshift_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"
echo "Commit   : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

TRAIN_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_armA_medshift.json"
DIMS_JSON="$EXPERIMENT_DIR/data/processed/sft/train_nisqa_llama_10k.json"
POLICY_BASE="$EXPERIMENT_DIR/models/sft_gc_timelast_timeaudio_h100"
HUB_REPO="Leng2beat/dpo-temporal-armA-medshift"

for f in "$TRAIN_JSON" "$DIMS_JSON"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing input: $f"; exit 1; fi
done
if [ ! -d "$POLICY_BASE" ]; then
    echo "ERROR: ARM A policy not found at $POLICY_BASE"
    echo "Restore: hf download Leng2beat/sft-gc-timelast-timeaudio --local-dir $POLICY_BASE"
    exit 1
fi

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "dpo_temporal_armA_medshift" \
    --model-id "$POLICY_BASE" \
    --json-path "$TRAIN_JSON" \
    --dims-source-json "$DIMS_JSON" \
    --data-root data \
    --batch-size 2 \
    --epochs 1 \
    --lr 1e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --hub-model-id "$HUB_REPO" \
    --no-hub-private \
    --final-save-only \
    --wandb-run-name "dpo-temporal-armA-medshift"

echo "=========================================="
echo "DPO temporal ARM A MEDIUM-SHIFT (0.75s) training complete: $(date)"
echo "Hub: $HUB_REPO"
echo "=========================================="
