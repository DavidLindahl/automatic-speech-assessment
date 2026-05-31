#!/bin/bash
### ============================================================
### DTU HPC — DPO retrain on the Full-SFT-PAIRED dataset, LR 1e-6
### Submit with: bsub < jobs/train/dpo_full_sft_paired_lr1e6.sh
###
### Paper-faithful follow-up to dpo_full_sft_lr1e6 (28517231): same
### hyperparameters (LR 1e-6, beta 0.4, 1 epoch, batch 2, grad-accum
### 16, ds_zero2), but the DPO dataset is now generated FROM the same
### Full SFT base it is about to be trained against
### (train_dpo_full_sft.json instead of the reused
### train_dpo_paper_half_h100_clean.json). Closes the one open caveat:
### rejected captions are now from the actual policy's own sampling
### distribution, not from an older policy.
###
### Question: does the dataset asymmetry explain the previous DPO
### regression (MAE +0.08 to +0.18 vs SFT base)? If MAE matches or
### beats the SFT base after this retrain -- yes, paper-faithfulness
### was the bug. If it still regresses -- the issue is structural
### (reference model choice, base saturation, etc.) and we name it
### honestly in the discussion.
###
### Reference model stays Qwen/Qwen2-7B (dpo-finetune.py default).
### NO --save-intermediate. NO Hub push. Local final-only save (~16 GB).
### Mirrors the working DPO mem profile: -n 4 x rusage[mem=64GB] = 256 GB.
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-full-sft-paired-lr1e6
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_full_sft_paired_lr1e6_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_full_sft_paired_lr1e6_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"

TRAIN_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_full_sft.json"

if [ ! -f "$TRAIN_JSON" ]; then
    echo "ERROR: missing Full-SFT-paired DPO dataset: $TRAIN_JSON"
    echo "Build it first via: bsub < jobs/train/generate_dpo_full_sft.sh"
    exit 1
fi

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "$EXPERIMENT_DIR/models/dpo_full_sft_paired_lr1e6" \
    --model-id "$EXPERIMENT_DIR/models/sft_full_paper_h100" \
    --json-path "$TRAIN_JSON" \
    --data-root data \
    --batch-size 2 \
    --epochs 1 \
    --lr 1e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --wandb-run-name "dpo-full-sft-paired-lr1e6"

echo "=========================================="
echo "DPO Full-SFT-paired LR 1e-6 training complete: $(date)"
echo "=========================================="
