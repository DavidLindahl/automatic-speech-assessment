#!/bin/bash
### ============================================================
### DTU HPC — FULL-target DPO on ARM A (unfactorized: all factors at once).
### Submit with: bsub < jobs/temporal/alld/dpo_temporal_armA_full.sh
###
### The unfactorized counterpart to the single-factor cycles. The rejected answer
### is the stage-1 model's RAW sampled output (train_dpo_armA_sampled.json, 13,495
### pairs), so caption, MOS, and the interval all differ from the gold chosen at
### once. This is the natural "full DPO" baseline and the discriminator for the
### key question: the three single-factor cycles stayed stable while the graded
### jitter set collapsed; is FACTORIZATION (one axis at a time) what kept them
### stable, or does the full multi-factor sample collapse the way jitter did?
###
###   - policy = ARM A (sft_gc_timelast_timeaudio_h100). TimeAudio subclass
###     auto-selected so abs_time_embedding survives + keeps training.
###   - --dims-source-json: temporal records carry mos but not noi/col/loud; the
###     ALLD reference (text-expert) prompt joins the full quality palette back
###     from the caption file by degraded-filename basename. REQUIRED.
###   - AUDIO: requires commit 51380b9 (collator feeds input_features +
###     feature_attention_mask). Earlier DPO ran the policy text-only.
###
### Same hyperparameters as every other temporal arm: LR 1e-6, beta 0.4, 1 epoch,
### batch 2, grad-accum 16. --final-save-only: one model saved at the end and
### pushed to the public Hub (save_only_model ~16 GB, OSError->Hub-rescue).
### Mem: -n 4 x rusage[mem=64GB] = 256 GB total, fits gpuh100 (~720 GB).
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-temporal-armA-full
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_armA_full_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_armA_full_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"
echo "Commit   : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

TRAIN_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_armA_sampled.json"
DIMS_JSON="$EXPERIMENT_DIR/data/processed/sft/train_nisqa_llama_10k.json"
POLICY_BASE="$EXPERIMENT_DIR/models/sft_gc_timelast_timeaudio_h100"
HUB_REPO="Leng2beat/dpo-temporal-armA-full"

for f in "$TRAIN_JSON" "$DIMS_JSON"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing input: $f"; exit 1; fi
done
if [ ! -d "$POLICY_BASE" ]; then
    echo "ERROR: ARM A policy not found at $POLICY_BASE"
    echo "Restore: hf download Leng2beat/sft-gc-timelast-timeaudio --local-dir $POLICY_BASE"
    exit 1
fi

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "dpo_temporal_armA_full" \
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
    --wandb-run-name "dpo-temporal-armA-full"

echo "=========================================="
echo "DPO temporal ARM A FULL (LR 1e-6) training complete: $(date)"
echo "Hub: $HUB_REPO"
echo "=========================================="
