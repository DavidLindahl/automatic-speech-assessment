#!/bin/bash
### ============================================================
### DTU HPC — Timestamp-cycle DPO on ARM A, SAMPLED source (arm 2 of the A/B).
### Submit with: bsub < jobs/temporal/alld/dpo_temporal_armA_hardneg_ts_iou06_nonorm.sh
###
### Phase-2 cyclic DPO, timestamp cycle. This is the SAMPLED arm of the 2-way
### A/B: rejected = ARM A's gold caption-last target with the gold caption and
### MOS held word-for-word fixed and ONLY the interval replaced by the policy's
### own sampled (wrong) interval (train_dpo_armA_hardneg_ts_iou06.json,
### 12,321 pairs). The JITTER arm (dpo_temporal_armA_jitter.sh) is the complement
### that uses graded hand-set shifts instead of the model's own errors. We run
### them sequentially (jitter first) to stay inside the /work3 quota, eval each
### full panel, and keep the winner as the timestamp-cycle policy.
###
###   - policy = ARM A (sft_gc_timelast_timeaudio_h100, t-IoU 0.88). It carries
###     the TimeAudio mechanisms (use_abs_time_embedding/use_time_tokens), so
###     dpo-finetune.py auto-selects the Qwen2AudioTimeForConditionalGeneration
###     subclass and the abs_time_embedding survives + keeps training.
###   - --dims-source-json: temporal records carry mos but not noi/col/loud, so
###     the ALLD reference (text-expert) prompt joins the full quality palette
###     back from the caption file by degraded-filename basename. REQUIRED for
###     temporal DPO (memo line 64); without it the reference prompt errors.
###   - AUDIO: requires commit 51380b9 (collator feeds input_features +
###     feature_attention_mask). Earlier DPO ran the policy text-only.
###
### Same hyperparameters as the working MOS DPO (dpo_full_sft_paired): LR 1e-6,
### beta 0.4, 1 epoch, batch 2, grad-accum 16. --final-save-only: NO intermediate
### checkpoints, exactly one model saved at the end and pushed to the public Hub
### (save_only_model keeps it ~16 GB, end save wrapped in OSError->Hub-rescue so
### the single-save path can't hang/overflow silently). Mem: -n 4 x
### rusage[mem=64GB] = 256 GB total, fits gpuh100 (~720 GB).
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-temporal-armA-hardneg-ts-iou06-nonorm
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_armA_hardneg_ts_iou06_nonorm_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_armA_hardneg_ts_iou06_nonorm_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"
echo "Commit   : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

TRAIN_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_armA_hardneg_ts_iou06.json"
DIMS_JSON="$EXPERIMENT_DIR/data/processed/sft/train_nisqa_llama_10k.json"
POLICY_BASE="$EXPERIMENT_DIR/models/sft_gc_timelast_timeaudio_h100"
HUB_REPO="Leng2beat/dpo-temporal-armA-hardneg-ts-iou06-nonorm"

for f in "$TRAIN_JSON" "$DIMS_JSON"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing input: $f"; exit 1; fi
done
if [ ! -d "$POLICY_BASE" ]; then
    echo "ERROR: ARM A policy not found at $POLICY_BASE"
    echo "Restore: hf download Leng2beat/sft-gc-timelast-timeaudio --local-dir $POLICY_BASE"
    exit 1
fi

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "dpo_temporal_armA_hardneg_ts_iou06_nonorm" \
    --model-id "$POLICY_BASE" \
    --json-path "$TRAIN_JSON" \
    --dims-source-json "$DIMS_JSON" \
    --data-root data \
    --batch-size 2 \
    --epochs 4 \
    --lr 1e-6 \
    --beta 0.4 \
    --no-length-norm \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --hub-model-id "$HUB_REPO" \
    --no-hub-private \
    --final-save-only \
    --wandb-run-name "dpo-temporal-armA-hardneg-ts-iou06-nonorm"

echo "=========================================="
echo "DPO temporal ARM A HARDNEG TS iou<0.6, 4ep (LR 1e-6) training complete: $(date)"
echo "Hub: $HUB_REPO"
echo "=========================================="
