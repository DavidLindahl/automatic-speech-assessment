#!/bin/bash
#BSUB -q milan
#BSUB -J expc-prep-clean
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 6:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/expc_prep_clean_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/expc_prep_clean_%J.err
set -euo pipefail
cd /work3/s234817/automatic-speech-assessment
source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH=src
echo "Start: $(date)"
python scripts/data/preprocess_clean_refs.py
echo "Done: $(date)"
