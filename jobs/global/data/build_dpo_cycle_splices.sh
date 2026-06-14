#!/bin/sh
### ============================================================
### DTU HPC — Splice ARM A's self-sample into the three DPO cycle datasets.
### Pure string surgery (no model inference), so a fast CPU job on the hpc
### queue. Reads train_dpo_armA_sampled.json (job 28647628) and writes three
### single-factor preference sets for the factorized cyclic DPO on ARM A:
###   - train_dpo_armA_cycle_mos.json              (sampled MOS only)
###   - train_dpo_armA_cycle_caption.json          (sampled caption + its MOS)
###   - train_dpo_armA_cycle_timestamp_sampled.json (sampled timestamps only)
### Each cycle's rejected is the model's own sampled mistake on that one factor.
### Submit with: bsub < jobs/global/data/build_dpo_cycle_splices.sh
### ============================================================

#BSUB -q hpc
#BSUB -J build-dpo-cycle-splices
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -W 1:00
#BSUB -o logs/build_dpo_cycle_splices_%J.out
#BSUB -e logs/build_dpo_cycle_splices_%J.err

set -eu

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs data/processed/dpo
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=src

echo "=========================================="
echo "Job ID   : ${LSB_JOBID:-local}"
echo "Host     : $(hostname)"
echo "Started  : $(date)"
echo "=========================================="

uv run python scripts/data/build_dpo_cycle_splices.py \
    --input-json data/processed/dpo/train_dpo_armA_sampled.json \
    --out-mos data/processed/dpo/train_dpo_armA_cycle_mos.json \
    --out-caption data/processed/dpo/train_dpo_armA_cycle_caption.json \
    --out-timestamp data/processed/dpo/train_dpo_armA_cycle_timestamp_sampled.json

echo "=========================================="
echo "DPO cycle splice complete: $(date)"
echo "=========================================="
