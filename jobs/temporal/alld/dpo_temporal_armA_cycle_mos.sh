#!/bin/bash
### ============================================================
### DTU HPC — MOS-cycle DPO on ARM A (optional, run last).
### Submit with: bsub < jobs/temporal/alld/dpo_temporal_armA_cycle_mos.sh
###
### Phase-2 cyclic DPO, MOS cycle. The rejected answer holds the gold caption
### and the gold interval fixed and changes ONLY the single MOS number to the
### policy's own sampled (wrong) value (train_dpo_armA_cycle_mos.json, 11,520
### pairs). This isolates the quality-score axis: it pushes the model toward the
### right MOS without touching the timestamps or the description. Marked optional
### in the plan; run after the timestamp A/B and the caption cycle if quota and
### time allow.
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
#BSUB -J dpo-temporal-armA-cycle-mos
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_armA_cycle_mos_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_armA_cycle_mos_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"
echo "Commit   : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

TRAIN_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_armA_cycle_mos.json"
DIMS_JSON="$EXPERIMENT_DIR/data/processed/sft/train_nisqa_llama_10k.json"
POLICY_BASE="$EXPERIMENT_DIR/models/sft_gc_timelast_timeaudio_h100"
HUB_REPO="Leng2beat/dpo-temporal-armA-cycle-mos"

for f in "$TRAIN_JSON" "$DIMS_JSON"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing input: $f"; exit 1; fi
done
if [ ! -d "$POLICY_BASE" ]; then
    echo "ERROR: ARM A policy not found at $POLICY_BASE"
    echo "Restore: hf download Leng2beat/sft-gc-timelast-timeaudio --local-dir $POLICY_BASE"
    exit 1
fi

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "dpo_temporal_armA_cycle_mos" \
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
    --wandb-run-name "dpo-temporal-armA-cycle-mos"

echo "=========================================="
echo "DPO temporal ARM A MOS cycle (LR 1e-6) training complete: $(date)"
echo "Hub: $HUB_REPO"
echo "=========================================="
