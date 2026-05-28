#!/bin/bash
### ============================================================
### DTU HPC — DPO on the new Full SFT base, LR 1e-6, 1x H100
### Submit with: bsub < jobs/train/dpo_full_sft_lr1e6.sh
###
### Exploratory: same DPO config as the working
### dpo_paper_half_h100_lr1e6_delimiterfix (LR 1e-6, beta 0.4, 1 epoch,
### batch 2, grad-accum 16, ds_zero2), only the policy warm-start
### changes -- from sft_warmup_paper_half_h100 (5k MOS captions, MAE
### 1.12-1.53) to sft_full_paper_h100 (10k MOS captions, MAE 0.27-0.38).
### Reference model stays Qwen/Qwen2-7B (the paper-faithful expert text
### LLM), per dpo-finetune.py default.
###
### Reuses the existing train_dpo_paper_half_h100_clean.json -- chosen
### captions are ground-truth from train_nisqa_llama_10k, only the
### rejected samples were drawn from the half-SFT. Strictly this is a
### slight asymmetry vs a paper-faithful pipeline (which would regen
### rejected from the new Full SFT), but the dataset is good enough for
### a "does Full SFT base help DPO at all" probe. If the result is
### interesting we can regen and retrain.
###
### Memory: -n 4 x rusage[mem=64GB] = 256 GB total. H100 nodes have
### ~720 GB system memory. Working DPO peaked at 87 GB.
### NO --save-intermediate (no /work3 quota overrun).
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-full-sft-lr1e6
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_full_sft_lr1e6_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_full_sft_lr1e6_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "$EXPERIMENT_DIR/models/dpo_full_sft_lr1e6" \
    --model-id "$EXPERIMENT_DIR/models/sft_full_paper_h100" \
    --json-path "$EXPERIMENT_DIR/data/processed/dpo/train_dpo_paper_half_h100_clean.json" \
    --data-root data \
    --batch-size 2 \
    --epochs 1 \
    --lr 1e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --wandb-run-name "dpo-full-sft-lr1e6"

echo "=========================================="
echo "DPO Full-SFT LR 1e-6 training complete: $(date)"
echo "=========================================="
