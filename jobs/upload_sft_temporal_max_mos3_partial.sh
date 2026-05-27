#!/bin/bash
### ============================================================
### DTU HPC — Upload SFT Temporal max_mos3 (PARTIAL, killed at step 305) to HF
### Submit with: bsub < jobs/upload_sft_temporal_max_mos3_partial.sh
###
### Note: this is NOT a finished model. The H100 run died mid-epoch-1
### at step 305/610. This pushes the partial checkpoint for archival only.
### ============================================================

#BSUB -q hpc
#BSUB -J upload-sft-temporal-max-mos3-partial
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 04:00
#BSUB -o logs/upload_sft_temporal_max_mos3_partial_%J.out
#BSUB -e logs/upload_sft_temporal_max_mos3_partial_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
source .venv/bin/activate

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Started  : $(date)"
echo "=========================================="

python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.upload_large_folder(
    folder_path="./models/sft_temporal_max_mos3/checkpoint-305",
    repo_id="Leng2beat/speech-quality-assessement-qwen2audio-sft-temporal-max-mos3-partial-step305",
    repo_type="model",
    num_workers=8,
)
PY

echo "Upload complete: $(date)"
