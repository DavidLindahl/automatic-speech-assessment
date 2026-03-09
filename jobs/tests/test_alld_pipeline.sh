#!/bin/sh
### ============================================================
### DTU HPC LSF job script — Test ALLD Pipeline
### Submit with: bsub < jobs/tests/test_alld_pipeline.sh
### ============================================================

### -- Queue: Standard CPU queue (No GPU needed for tokenizer testing) --
#BSUB -q hpc

### -- Job name --
#BSUB -J test-alld-pipeline

### -- CPU cores --
#BSUB -n 1

### -- System memory --
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB

### -- Walltime (15 minutes is more than enough) --
#BSUB -W 0:15

### -- Output / error logs --
#BSUB -o logs/test_alld_%J.out
#BSUB -e logs/test_alld_%J.err

set -euo pipefail

# Make sure this path matches your actual project directory
PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs

# Activate your python environment
source .venv/bin/activate

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Started  : $(date)"
echo "=========================================="

# Fix the module import issue explicitly for the HPC environment
export PYTHONPATH=src

# Run the test script
uv run python tests/test_alld_pipeline.py

echo "=========================================="
echo "Test complete: $(date)"
echo "=========================================="
