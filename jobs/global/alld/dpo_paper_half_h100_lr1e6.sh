#!/bin/bash
### ============================================================
### DTU HPC — DPO/ALLD paper-half warmup, LR 1e-6 ablation, 1x H100
### Submit with: bsub < jobs/global/alld/dpo_paper_half_h100_lr1e6.sh
###
### One-variable ablation of dpo_paper_half_h100.sh: LR 5e-6 -> 1e-6.
### The 5e-6 run (28481644) collapsed onto the EOS token (diag 28483665);
### the only DPO run that ever produced real captions (April 27, dsat5nxi)
### used LR 1e-6. Everything else is held identical: same SFT base, same
### dataset, same beta, epochs, batch, grad-accum. Distinct output dir,
### job name, W&B run name and Hub repo so the 5e-6 artifacts are untouched.
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-paper-half-h100-lr1e6
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_paper_half_h100_lr1e6_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_paper_half_h100_lr1e6_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh

# Distinct Hub repo so the 5e-6 run's repo is untouched.
HUB_MODEL_ID="${HUB_MODEL_ID:-Leng2beat/speech-quality-assessement-qwen2audio-dpo-paper-half-lr1e6}"

torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "$EXPERIMENT_DIR/models/dpo_paper_half_h100_lr1e6" \
    --model-id "$EXPERIMENT_DIR/models/sft_warmup_paper_half_h100" \
    --json-path "$EXPERIMENT_DIR/data/processed/dpo/train_dpo_paper_half_h100_clean.json" \
    --data-root data \
    --batch-size 2 \
    --epochs 1 \
    --lr 1e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --wandb-run-name "dpo-paper-half-h100-lr1e6" \
    --hub-model-id "$HUB_MODEL_ID" \
    --save-steps 200 \
    --save-total-limit 1

echo "=========================================="
echo "DPO training complete: $(date)"
echo "=========================================="
