#!/bin/bash
### ============================================================
### DTU HPC — Upload completed SFT Temporal max_mos3 model to HF
### Submit with: bsub < jobs/upload_sft_temporal_max_mos3.sh
### ============================================================

#BSUB -q hpc
#BSUB -J upload-sft-temporal-max-mos3
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 04:00
#BSUB -o logs/upload_sft_temporal_max_mos3_%J.out
#BSUB -e logs/upload_sft_temporal_max_mos3_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
source .venv/bin/activate

REPO_ID="${HF_REPO_ID:-Leng2beat/SFT-temporal}"
MODEL_DIR="${MODEL_DIR:-./models/sft_temporal}"

echo "=========================================="
echo "Job ID    : ${LSB_JOBID:-local}"
echo "Host      : $(hostname)"
echo "Started   : $(date)"
echo "Repo ID   : $REPO_ID"
echo "Model dir : $MODEL_DIR"
echo "=========================================="

python - <<PY
from huggingface_hub import HfApi

repo_id = "${REPO_ID}"
folder_path = "${MODEL_DIR}"

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
api.upload_large_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model",
    num_workers=8,
)
PY

echo "Upload complete: $(date)"
