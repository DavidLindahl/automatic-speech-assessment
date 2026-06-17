#!/bin/bash
### ============================================================
### DTU HPC — MOS-cycle DPO on ARM A, HIGHER BETA (improvement experiment).
### Submit with: bsub < jobs/temporal/alld/dpo_temporal_armA_cycle_mos_beta1.sh
###
### Follow-up to the cycle program. All 4 model-sample cycles saturated (stable,
### none beat SFT) with tiny reward margins, the fingerprint of a preference
### signal too weak to move the policy off the reference. This arm strengthens
### that push on the BEST-behaved cycle: the MOS cycle (lowest MOS-MAE 0.45,
### highest BLEU 23.1, localization preserved at 0.874). The ONLY change vs the
### baseline MOS cycle (28654286) is beta 0.4 -> 1.0; same data, policy, recipe.
### Question: does a harder preference push move the MOS cycle past SFT
### (MOS-MAE 0.40), or does it over-optimize/collapse (the jitter lesson)? Either
### is informative: improvement = signal was too weak; collapse = a real ceiling.
###
###   - policy = ARM A (sft_gc_timelast_timeaudio_h100). TimeAudio subclass
###     auto-selected so abs_time_embedding survives + keeps training.
###   - --dims-source-json: REQUIRED (temporal records carry mos but not
###     noi/col/loud; the ALLD reference prompt joins the full palette back).
###   - AUDIO: requires commit 51380b9 (collator feeds input_features +
###     feature_attention_mask).
###
### LR 1e-6, BETA 1.0 (was 0.4), 1 epoch, batch 2, grad-accum 16. --final-save-only
### to the public Hub. Mem: -n 4 x rusage[mem=64GB] = 256 GB total, fits gpuh100.
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-temporal-armA-cycle-mos-beta1
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_armA_cycle_mos_beta1_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_armA_cycle_mos_beta1_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"
echo "Commit   : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

TRAIN_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_armA_cycle_mos.json"
DIMS_JSON="$EXPERIMENT_DIR/data/processed/sft/train_nisqa_llama_10k.json"
POLICY_BASE="$EXPERIMENT_DIR/models/sft_gc_timelast_timeaudio_h100"
HUB_REPO="Leng2beat/dpo-temporal-armA-cycle-mos-beta1"

for f in "$TRAIN_JSON" "$DIMS_JSON"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing input: $f"; exit 1; fi
done
if [ ! -d "$POLICY_BASE" ]; then
    echo "ERROR: ARM A policy not found at $POLICY_BASE"
    echo "Restore: hf download Leng2beat/sft-gc-timelast-timeaudio --local-dir $POLICY_BASE"
    exit 1
fi

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "dpo_temporal_armA_cycle_mos_beta1" \
    --model-id "$POLICY_BASE" \
    --json-path "$TRAIN_JSON" \
    --dims-source-json "$DIMS_JSON" \
    --data-root data \
    --batch-size 2 \
    --epochs 1 \
    --lr 1e-6 \
    --beta 1.0 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --hub-model-id "$HUB_REPO" \
    --no-hub-private \
    --final-save-only \
    --wandb-run-name "dpo-temporal-armA-cycle-mos-beta1"

echo "=========================================="
echo "DPO temporal ARM A MOS cycle BETA 1.0 training complete: $(date)"
echo "Hub: $HUB_REPO"
echo "=========================================="
