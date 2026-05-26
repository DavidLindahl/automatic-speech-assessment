#!/bin/bash
### ============================================================
### DTU HPC — Paper-faithful Full SFT on full 10k MOS examples, 1x H100
### Mirrors sft_warmup_paper_half_h100.sh shape but drops --max-samples
### to use all 10k captions. Fills the paper Table 1 "Full-ft" baseline.
### Local-only save (no HF Hub); final ~16 GB checkpoint to /work3.
### Submit with: bsub < jobs/sft/sft_full_paper_h100.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-full-paper-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -o logs/sft_full_paper_h100_%J.out
#BSUB -e logs/sft_full_paper_h100_%J.err

source "$(dirname "$0")/../_lib/preamble.sh"

torchrun --nproc_per_node=1 scripts/train/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --json-path data/processed/sft/train_nisqa_llama_10k.json \
    --data-root data \
    --model-name "$EXPERIMENT_DIR/models/sft_full_paper_h100" \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 2 \
    --lr 1e-5 \
    --val-split 0 \
    --wandb-run-name "sft-full-paper-h100"

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
