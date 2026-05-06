#!/bin/bash
### ============================================================
### DTU HPC — Inspect ALLDDPOCollator labels on a real batch
### Submit with: bsub < jobs/diagnose_dpo_label_mask.sh
###
### CPU-only diagnostic. Loads the SFT processor + DPO dataset,
### runs the actual ALLDDPOCollator on N=2 features, and reports
### whether the label mask aligns with the response region or
### overlaps the prompt (which would confirm the bug).
### ============================================================

#BSUB -q hpc
#BSUB -J diag-dpo-labelmask
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 00:15
#BSUB -o logs/diag_dpo_labelmask_%J.out
#BSUB -e logs/diag_dpo_labelmask_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
source .venv/bin/activate

export PYTHONUNBUFFERED=1

# HF cache off /work3 (Qwen2-7B reference tokenizer download).
if [ -d "/scratch" ] && [ -w "/scratch" ]; then
    export HF_HOME="/scratch/$USER/hf_cache"
elif [ -w "/tmp" ]; then
    export HF_HOME="/tmp/$USER/hf_cache"
else
    export HF_HOME="$PROJECT_DIR/.cache/huggingface"
fi
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

echo "=========================================="
echo "Job ID  : $LSB_JOBID"
echo "Started : $(date)"
echo "=========================================="

uv run python scripts/diagnose_dpo_label_mask.py \
    --sft-model models/sft_warmup_paper_half_h100 \
    --ref-model Qwen/Qwen2-7B \
    --json-path data/processed/train_dpo_paper_half_h100_clean.json \
    --data-root data \
    --n 2

echo "Done: $(date)"
