#!/bin/sh
### ============================================================
### DTU HPC — Prepare temporal smoke-test dataset (CPU)
### ============================================================

#BSUB -q hpc
#BSUB -J prep-temporal-smoke
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 2:00
#BSUB -o logs/prepare_temporal_smoke_%J.out
#BSUB -e logs/prepare_temporal_smoke_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Started  : $(date)"
echo "=========================================="

uv run python scripts/data/prepare_temporal_smoke.py \
    --train-samples 1000 \
    --test-samples 200

echo "Temporal smoke dataset preparation complete: $(date)"
