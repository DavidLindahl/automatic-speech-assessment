#!/bin/bash
### ============================================================
### DTU HPC — ALLD/DPO on the gc-PLAIN SFT model
### Aligns the Setup-1 plain global-caption SFT model (28615748) with the
### ALLD dual-stream DPO loss against pairs from generate_dpo_gc_plain.sh
### (chosen = gold temporal target, rejected = the model's own collapsed
### sample). Goal: push the policy off its collapsed near-constant interval
### toward the true per-clip timestamp.
###
### Known-good DPO config (from dpo_full_sft_paired_lr1e6, length-norm ON per
### our ablation): lr 1e-6, beta 0.4, 1 epoch, batch 2, grad-accum 16, ds_zero2.
### NO --hub-model-id and NO --save-intermediate => save_strategy="no",
### final-only LOCAL save (~16 GB), quota-safe. Push to Hub manually after.
###
### Mem (per-core rusage rule): rusage[mem=64GB] x -n 4 = 256 GB total, within
### the gpuh100 ~720 GB node. Matches the working DPO mem profile.
### Submit with: bsub < jobs/train/dpo_gc_plain.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-gc-plain
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_gc_plain_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_gc_plain_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"

TRAIN_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_gc_plain.json"

if [ ! -f "$TRAIN_JSON" ]; then
    echo "ERROR: missing gc-plain DPO dataset: $TRAIN_JSON"
    echo "Build it first via: bsub < jobs/train/generate_dpo_gc_plain.sh"
    exit 1
fi

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "$EXPERIMENT_DIR/models/dpo_gc_plain_h100" \
    --model-id "$EXPERIMENT_DIR/models/sft_temporal_gc_plain_h100" \
    --json-path "$TRAIN_JSON" \
    --data-root data \
    --dims-source-json data/processed/sft/train_nisqa_llama_10k.json \
    --batch-size 2 \
    --epochs 1 \
    --lr 1e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --wandb-run-name "dpo-gc-plain"

echo "=========================================="
echo "DPO gc-plain training complete: $(date)"
echo "=========================================="
