#!/bin/bash
### ============================================================
### DTU HPC — ALLD CYCLE 2: DPO on the cycle-1 best model, LR 1e-6
### Submit with: bsub < jobs/train/dpo_cycle2.sh
###
### Second iterative ALLD round. Identical hyperparameters to cycle 1
### (dpo_full_sft_paired_lr1e6: LR 1e-6, beta 0.4, 1 epoch, batch 2,
### grad-accum 16, ds_zero2) so the cycle-1-vs-cycle-2 comparison is a
### clean single-axis change. The two things that differ from cycle 1:
###   1. --model-id  -> the cycle-1 DPO model (models/dpo_full_sft_paired_lr1e6,
###      Hub Leng2beat/DPO_Global_Full), NOT the SFT. This is the new policy
###      warm-start.
###   2. --json-path -> train_dpo_cycle2.json, whose rejected captions were
###      resampled from that same DPO model (generate_dpo_cycle2.sh).
###
### ALLD reference model stays Qwen/Qwen2-7B (dpo-finetune.py default,
### --ref-model-id). It is the frozen cross-modal distillation signal and is
### deliberately NOT coupled to --model-id, so this remains ALLD, not
### standard previous-policy-reference iterative DPO.
###
### Question: does a 2nd ALLD cycle compound the cycle-1 gains, or does the
### policy regress (over-optimization / diversity collapse) because it is
### already near the ALLD fixed point? Either outcome is a clean result.
###
### NO --save-intermediate. NO Hub push. Local final-only save (~16 GB).
### Mem profile mirrors the working DPO: -n 4 x rusage[mem=64GB] = 256 GB
### total, fits gpuh100 (~720 GB/node). Audited.
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-cycle2
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_cycle2_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_cycle2_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"

TRAIN_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_cycle2.json"
POLICY_BASE="$EXPERIMENT_DIR/models/dpo_full_sft_paired_lr1e6"

if [ ! -f "$TRAIN_JSON" ]; then
    echo "ERROR: missing cycle-2 DPO dataset: $TRAIN_JSON"
    echo "Build it first via: bsub < jobs/train/generate_dpo_cycle2.sh"
    exit 1
fi
if [ ! -d "$POLICY_BASE" ]; then
    echo "ERROR: cycle-1 DPO policy base not found at $POLICY_BASE"
    echo "Download it first: hf download Leng2beat/DPO_Global_Full --local-dir $POLICY_BASE"
    exit 1
fi

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "$EXPERIMENT_DIR/models/dpo_cycle2" \
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
    --wandb-run-name "dpo-cycle2"

echo "=========================================="
echo "DPO cycle-2 (LR 1e-6) training complete: $(date)"
echo "=========================================="
