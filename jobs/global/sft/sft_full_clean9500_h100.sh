#!/bin/bash
### ============================================================
### DTU HPC - Clean-split Full SFT on the disjoint 9500 in-domain split, 1x H100
### Identical to sft_full_paper_h100.sh except the training JSON points at the
### 9500-clip split (train_nisqa_llama_indomain.json) instead of the full 10k
### file. This removes the in-domain test leakage: the full-10k file contained
### all 500 in-domain test clips, so the reported in-domain SFT numbers were
### optimistic. This run never sees the 500-clip test set.
### Local-only save (no HF Hub); final ~16 GB checkpoint to /work3.
### Submit with: bsub < jobs/global/sft/sft_full_clean9500_h100.sh
###
### Memory audit (LSF rusage[mem] is PER-CORE): 48GB x 4 cores = 192 GB total,
### fits gpuh100 (~720 GB/node). Identical to the approved sft_full_paper_h100.sh.
### ============================================================

#BSUB -q gpuh100
#BSUB -J sft-full-clean9500-h100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 24:00
#BSUB -o logs/sft_full_clean9500_h100_%J.out
#BSUB -e logs/sft_full_clean9500_h100_%J.err

source /work3/s234817/automatic-speech-assessment/jobs/_lib/preamble.sh

torchrun --nproc_per_node=1 scripts/train/supervised-finetune.py \
    --model-id Qwen/Qwen2-Audio-7B \
    --json-path data/processed/global/sft/train_nisqa_llama_indomain.json \
    --data-root data \
    --model-name "$EXPERIMENT_DIR/models/sft_full_clean9500_h100" \
    --bf16 \
    --deepspeed configs/ds_zero2.json \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --epochs 2 \
    --lr 1e-5 \
    --val-split 0 \
    --wandb-run-name "sft-full-clean9500-h100"

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="
