#!/bin/sh
### ============================================================
### DTU HPC — Generate temporal-factor (pair-type A) DPO data
### The localization-pressure half of the Phase-2 factorized temporal DPO
### (research/temporal-loss-design.md). chosen = gold target, rejected =
### gold caption with ONLY the interval jittered (graded 0.5/1/2/4 s). No
### model inference, pure string surgery, so this is a CPU job on the hpc
### queue (no GPU, fast). Output train_dpo_gc_temporal_factor.json, mixed
### ~1:1 with the existing pair-type B set (train_dpo_gc_plain.json) at the
### DPO training stage.
### Submit with: bsub < jobs/temporal/data/generate_dpo_temporal_factor.sh
### ============================================================

#BSUB -q hpc
#BSUB -J gen-dpo-temporal-factor
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -W 1:00
#BSUB -o logs/gen_dpo_temporal_factor_%J.out
#BSUB -e logs/gen_dpo_temporal_factor_%J.err

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

uv run python scripts/data/generate_dpo_temporal_factor.py \
    --input-json data/processed/dpo/train_dpo_gc_plain.json \
    --output-json data/processed/dpo/train_dpo_gc_temporal_factor.json \
    --offsets 0.5,1,2,4

echo "=========================================="
echo "Temporal-factor DPO data generation complete: $(date)"
echo "=========================================="
