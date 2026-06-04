#!/bin/bash
### ============================================================
### DTU HPC — DPO ABLATION: length-normalisation OFF
### Submit with: bsub < jobs/train/dpo_full_sft_paired_nonorm.sh
###
### Exact clone of the cycle-1 paired DPO run 28557823
### (dpo_full_sft_paired_lr1e6 -- the strongest DPO model that
### warm-starts from the Full SFT base), with ONE variable changed:
### the loss runs WITHOUT length normalisation (--no-length-norm),
### i.e. the raw summed-log-prob ALLD objective from before the
### 2026-04-14 mode-collapse fix (commit 988a640).
###
### Everything else is held identical to 28557823:
###   policy   = Leng2beat/SFT_Global_Full (the Full SFT base, on HF Hub).
###              VERIFIED identical to the deleted local sft_full_paper_h100
###              that 28557823 trained from: its training_args.bin carries
###              output_dir=.../models/sft_full_paper_h100, epochs 2, LR 1e-5,
###              batch 4 -- the canonical Full SFT (28504365). Loaded from Hub
###              because the local /work3 copy was retired under quota; the
###              preamble redirects HF_HOME to node-local scratch so the
###              ~16 GB download does not touch the /work3 or home quota.
###   dataset  = train_dpo_full_sft.json (rejected sampled from that SFT)
###   ref      = Qwen/Qwen2-7B (dpo-finetune.py default, frozen)
###   LR 1e-6, beta 0.4, 1 epoch, batch 2, grad-accum 16, ds_zero2.
###
### Purpose: isolate length normalisation. The original fix bundled it
### with a data-gen change, so its individual contribution was never
### measured. This run is the direct A/B counterpart to 28557823.
###   - If MAE/diversity match 28557823 -> length-norm was not load-bearing.
###   - If this collapses (e.g. the ' speech' x60 / EOS fingerprint) ->
###     length-norm is necessary in the cross-modal ALLD setup. Clean result.
###
### NO --save-intermediate. NO Hub push. Local final-only save (~16 GB).
### Mem profile mirrors 28557823: -n 4 x rusage[mem=64GB] = 256 GB total
### (fits gpuh100 ~720 GB/node).
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-full-sft-paired-nonorm
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_full_sft_paired_nonorm_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_full_sft_paired_nonorm_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"

TRAIN_JSON="$EXPERIMENT_DIR/data/processed/dpo/train_dpo_full_sft.json"

if [ ! -f "$TRAIN_JSON" ]; then
    echo "ERROR: missing Full-SFT-paired DPO dataset: $TRAIN_JSON"
    echo "Build it first via: bsub < jobs/train/generate_dpo_full_sft.sh"
    exit 1
fi

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "$EXPERIMENT_DIR/models/dpo_full_sft_paired_nonorm" \
    --model-id "Leng2beat/SFT_Global_Full" \
    --json-path "$TRAIN_JSON" \
    --data-root data \
    --batch-size 2 \
    --epochs 1 \
    --lr 1e-6 \
    --beta 0.4 \
    --no-length-norm \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --wandb-run-name "dpo-full-sft-paired-nonorm"

echo "=========================================="
echo "DPO Full-SFT-paired NO-length-norm training complete: $(date)"
echo "=========================================="
