#!/bin/bash
### ============================================================
### DTU HPC — Full DPO/ALLD run with the prompt-delimiter fix, 1x H100
### Submit with: bsub < jobs/global/alld/dpo_paper_half_h100_delimiterfix.sh
###
### First full DPO run after the EOS-collapse fix. PROMPT_TEMPLATE now ends
### with "\n" (commit a007248) so the first response word is a standalone
### supervised token. The 1024-sample delimiter smoke (28483856) confirmed
### the fix: P(EOS) at step 0 dropped from 0.64-0.79 to 0.0000, the model
### generates full captions again (diag 28484057).
###
### Hub push DISABLED: the Leng2beat HF account is out of private-repo
### storage, so push_to_hub 403s and leaves the checkpoint without its
### processor files. With no --hub-model-id the run does a single clean
### trainer.save_model() at the end (save_strategy="no"). All-or-nothing,
### but the prior full run 28481644 completed training in 1h46m with no
### intermediate save needed. Same setup.
###
### Memory audit (LSF rusage[mem] is PER-CORE): -n 4 x rusage[mem=64GB]
### = 256 GB total. H100 nodes have ~720 GB system memory, so this fits.
### ============================================================

#BSUB -q gpuh100
#BSUB -J dpo-paper-half-delimiterfix
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -M 64GB
#BSUB -W 24:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/dpo_paper_half_h100_delimiterfix_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/dpo_paper_half_h100_delimiterfix_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh
echo "Branch   : $(git branch --show-current 2>/dev/null || echo unknown)"

# Full DPO: paper settings (LR 5e-6, beta 0.4, 1 epoch) on the cleaned
# 9979-row dataset. The PROMPT_TEMPLATE "\n" delimiter fix is picked up
# automatically. No --hub-model-id -> push_to_hub disabled, single clean
# local save at the end.
torchrun --nproc_per_node=1 scripts/train/dpo-finetune.py \
    --model-name "$EXPERIMENT_DIR/models/dpo_paper_half_h100_delimiterfix" \
    --model-id "$EXPERIMENT_DIR/models/sft_warmup_paper_half_h100" \
    --json-path "$EXPERIMENT_DIR/data/processed/dpo/train_dpo_paper_half_h100_clean.json" \
    --data-root data \
    --batch-size 2 \
    --epochs 1 \
    --lr 5e-6 \
    --beta 0.4 \
    --gradient-accumulation-steps 16 \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --wandb-run-name "dpo-paper-half-h100-delimiterfix"

echo "=========================================="
echo "DPO training complete: $(date)"
echo "=========================================="
