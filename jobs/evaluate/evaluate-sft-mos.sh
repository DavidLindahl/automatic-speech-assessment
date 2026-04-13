#!/bin/sh
### ============================================================
### DTU HPC LSF job script — Model Evaluation
### Submit with: bsub < jobs/evaluate/evaluate.sh
### ============================================================

### -- Queue: L40S 48GB --
#BSUB -q gpua10

### -- Job name --
#BSUB -J evaluate-model

### -- CPU cores (min 4 per GPU) --
#BSUB -n 4

### -- Request 1 GPU --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- Single node --
#BSUB -R "span[hosts=1]"

### -- System memory (model loading ~15GB + overhead) --
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB

### -- Walltime (evaluation can take a while) --
#BSUB -W 4:00

### -- Output / error files --
#BSUB -o logs/evaluate_%J.out
#BSUB -e logs/evaluate_%J.err

# -- end of LSF options --

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p results/inference/sft

module load cuda/11.8 || true

source .venv/bin/activate

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi



# Define which model to evaluate
MODEL_PATH="models/sft_full"
OUTPUT_PATH="results/evaluation/sft_full_eval"

# Define test datasets (you can add more here)
DATASETS=(
    "data/processed/test_FOR.json"
    "data/processed/test_LIVE.json"
    "data/processed/test_P501.json"
)

echo "Evaluating datasets: ${DATASETS[*]}"
uv run python src/asa/evaluate.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_PATH" \
    --dataset-path "${DATASETS[0]}" \
    --dataset-path "${DATASETS[1]}" \
    --dataset-path "${DATASETS[2]}" \
    --batch-size 8

echo ""
echo "=========================================="
echo "Evaluation complete: $(date)"
echo "=========================================="
