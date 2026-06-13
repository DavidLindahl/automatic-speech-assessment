#!/bin/bash
### ============================================================
### DTU HPC — SMOKE: temporal DPO on ARM A (audio-path gate).
### Submit with: bsub < jobs/train/dpo_temporal_armA_smoke.sh
###
### Purpose: prove the fixed DPO pipeline trains the policy WITH audio on a GPU
### node before committing a 6h A/B. After commit 51380b9 the collator feeds
### input_features + feature_attention_mask; this runs a handful of real steps
### on ARM A (the TimeAudio subclass, auto-detected) over a tiny slice of the
### jitter set and must finish with a finite loss and non-trivial reward logs.
###
### Tiny by design: --max-samples 16, batch 2, grad-accum 1, 1 epoch => ~8
### steps. NO Hub push, NO checkpoint save (save_total_limit small + no
### --hub-model-id => save_strategy "no"), so it is quota-free and fast.
###
### Mem: -n 4 x rusage[mem=48GB] = 192 GB total, fits gpuh100 (~720 GB).
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-temporal-armA-smoke
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 1:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_armA_smoke_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_temporal_armA_smoke_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"
echo "Commit   : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

TRAIN_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_armA_jitter.json"
DIMS_JSON="$EXPERIMENT_DIR/data/processed/sft/train_nisqa_llama_10k.json"
POLICY_BASE="$EXPERIMENT_DIR/models/sft_gc_timelast_timeaudio_h100"

for f in "$TRAIN_JSON" "$DIMS_JSON"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing input: $f"; exit 1; fi
done
if [ ! -d "$POLICY_BASE" ]; then
    echo "ERROR: ARM A policy not found at $POLICY_BASE"
    echo "Restore: hf download Leng2beat/sft-gc-timelast-timeaudio --local-dir $POLICY_BASE"
    exit 1
fi

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "dpo_temporal_armA_smoke" \
    --model-id "$POLICY_BASE" \
    --json-path "$TRAIN_JSON" \
    --dims-source-json "$DIMS_JSON" \
    --data-root data \
    --max-samples 16 \
    --batch-size 2 \
    --epochs 1 \
    --lr 1e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 1 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --wandb-project "" \
    --wandb-run-name "dpo-temporal-armA-smoke"

echo "=========================================="
echo "DPO temporal ARM A SMOKE complete: $(date)"
echo "If you see a finite train_loss and reward logs above, the audio path is live."
echo "=========================================="
