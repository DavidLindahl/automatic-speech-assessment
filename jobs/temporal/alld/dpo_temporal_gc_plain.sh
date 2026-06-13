#!/bin/bash
### ============================================================
### DTU HPC — Plain DPO on the temporal SFT (gc-plain arm), LR 1e-6
### Submit with: bsub < jobs/temporal/alld/dpo_temporal_gc_plain.sh
###
### The straightforward "does DPO work on the temporal model" run, same
### recipe as the MOS DPO that worked (dpo_cycle2 / dpo_full_sft_paired):
###   - policy  = the plain temporal SFT (sft_temporal_gc_plain_h100),
###     restored from Leng2beat/sft-temporal-global-caption-plain. This arm
###     is VANILLA SFT (free-text <|seconds|> timestamps, NO time tokens, NO
###     abs-time embedding), so it loads as a plain Qwen2-Audio model and
###     needs no mechanism flags.
###   - preference data = train_dpo_gc_plain.json: chosen = gold temporal
###     target, rejected = this same SFT's own samples (temp 1.1, top-p 0.9).
###     Self-sampled, so this is the paper-faithful match.
###   - frozen ALLD reference stays Qwen/Qwen2-7B (dpo-finetune.py default).
###
### Same hyperparameters as the working MOS DPO: LR 1e-6, beta 0.4, 1 epoch,
### batch 2, grad-accum 16, ds_zero2. NO --save-intermediate, NO Hub push,
### local final-only save (~16 GB). Mem: -n 4 x rusage[mem=64GB] = 256 GB,
### audited, fits gpuh100.
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-temporal-gc-plain
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_gc_plain_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_gc_plain_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"

TRAIN_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_gc_plain.json"
POLICY_BASE="$EXPERIMENT_DIR/models/sft_temporal_gc_plain_h100"

if [ ! -f "$TRAIN_JSON" ]; then
    echo "ERROR: missing DPO dataset: $TRAIN_JSON"
    exit 1
fi
if [ ! -d "$POLICY_BASE" ]; then
    echo "ERROR: temporal SFT policy base not found at $POLICY_BASE"
    echo "Restore it: hf download Leng2beat/sft-temporal-global-caption-plain --local-dir $POLICY_BASE"
    exit 1
fi

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "$EXPERIMENT_DIR/models/dpo_temporal_gc_plain" \
    --model-id "$POLICY_BASE" \
    --json-path "$TRAIN_JSON" \
    --data-root data \
    --batch-size 2 \
    --epochs 1 \
    --lr 1e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --wandb-run-name "dpo-temporal-gc-plain"

echo "=========================================="
echo "DPO temporal gc-plain (LR 1e-6) training complete: $(date)"
echo "=========================================="
