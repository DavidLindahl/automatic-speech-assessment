#!/bin/bash
### ============================================================
### DTU HPC — Caption-cycle DPO on ARM A.
### Submit with: bsub < jobs/temporal/alld/dpo_temporal_armA_cycle_caption.sh
###
### Phase-2 cyclic DPO, CAPTION cycle. The rejected answer holds the gold
### interval and the gold MOS fixed and replaces ONLY the caption/head text with
### the policy's own sampled (wrong) description
### (train_dpo_armA_cycle_caption.json, 13,488 pairs). This isolates the
### description axis: it pushes the model toward better wording without touching
### the timestamps, so we can check whether aligning the caption helps the
### overall answer without eroding localization (t-IoU vs ARM A 0.88).
###
###   - policy = ARM A (sft_gc_timelast_timeaudio_h100). TimeAudio subclass
###     auto-selected so abs_time_embedding survives + keeps training.
###   - --dims-source-json: temporal records carry mos but not noi/col/loud; the
###     ALLD reference (text-expert) prompt joins the full quality palette back
###     from the caption file by degraded-filename basename. REQUIRED.
###   - AUDIO: requires commit 51380b9 (collator feeds input_features +
###     feature_attention_mask). Earlier DPO ran the policy text-only.
###
### Same hyperparameters as the working MOS DPO: LR 1e-6, beta 0.4, 1 epoch,
### batch 2, grad-accum 16. --final-save-only: exactly one model saved at the end
### and pushed to the public Hub (save_only_model ~16 GB, OSError->Hub-rescue).
### Mem: -n 4 x rusage[mem=64GB] = 256 GB total, fits gpuh100 (~720 GB).
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-temporal-armA-cycle-caption
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_armA_cycle_caption_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_armA_cycle_caption_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"
echo "Commit   : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

TRAIN_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_armA_cycle_caption.json"
DIMS_JSON="$EXPERIMENT_DIR/data/processed/sft/train_nisqa_llama_10k.json"
POLICY_BASE="$EXPERIMENT_DIR/models/sft_gc_timelast_timeaudio_h100"
HUB_REPO="Leng2beat/dpo-temporal-armA-cycle-caption"

for f in "$TRAIN_JSON" "$DIMS_JSON"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing input: $f"; exit 1; fi
done
if [ ! -d "$POLICY_BASE" ]; then
    echo "ERROR: ARM A policy not found at $POLICY_BASE"
    echo "Restore: hf download Leng2beat/sft-gc-timelast-timeaudio --local-dir $POLICY_BASE"
    exit 1
fi

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "dpo_temporal_armA_cycle_caption" \
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
    --wandb-run-name "dpo-temporal-armA-cycle-caption"

echo "=========================================="
echo "DPO temporal ARM A CAPTION cycle (LR 1e-6) training complete: $(date)"
echo "Hub: $HUB_REPO"
echo "=========================================="
