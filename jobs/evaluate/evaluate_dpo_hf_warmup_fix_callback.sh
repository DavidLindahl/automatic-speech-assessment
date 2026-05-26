#!/bin/sh
### ============================================================
### DTU HPC LSF job script - DPO checkpoint evaluation with callback
### Purpose: evaluate saved dpo_hf_warmup_fix checkpoint and wake Thor on exit
### ============================================================

#BSUB -q gpua10
#BSUB -J eval-dpo-hf-warmup-fix
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 4:00
#BSUB -o logs/evaluate_%J.out
#BSUB -e logs/evaluate_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"

mkdir -p logs
mkdir -p results/inference/dpo

RELAY_URL="https://mac-servers.tailccbbb9.ts.net:8443/dtu-job"
RELAY_TOKEN="2dc7b1b0df08dd349d957c77e230350bfa4fc8f86aefcb4d"
WORKFLOW="inspect-eval-results"
TODO_FILE="todos/eval_dpo_hf_warmup_fix.md"
RUN_FILE="runs/2026-04-13_eval_dpo_hf_warmup_fix_inference_28194908.md"
MODEL_PATH="models/dpo_hf_warmup_fix"
RESULTS_PATH="results/evaluation/dpo_hf_warmup_fix_eval"

notify_exit() {
  code="$1"
  status="success"
  if [ "$code" -ne 0 ]; then
    status="failed"
  fi

  payload=$(cat <<JSON
{"job_id":"${LSB_JOBID:-unknown}","job_name":"${LSB_JOBNAME:-eval-dpo-hf-warmup-fix}","status":"${status}","queue":"gpua10","host":"$(hostname)","workflow":"${WORKFLOW}","todo_file":"${TODO_FILE}","run_file":"${RUN_FILE}","model_path":"${MODEL_PATH}","results_path":"${RESULTS_PATH}"}
JSON
)

  curl --max-time 10 --fail --silent --show-error \
    -X POST "$RELAY_URL" \
    -H "Authorization: Bearer $RELAY_TOKEN" \
    -H "Content-Type: application/json" \
    -d "$payload" || true
}

trap 'code=$?; notify_exit "$code"' EXIT

module load cuda/11.8 || true
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi

OUTPUT_PATH="$RESULTS_PATH"

DATASETS=(
    "data/processed/test_FOR.json"
    "data/processed/test_LIVE.json"
    "data/processed/test_P501.json"
)

echo "Evaluating datasets: ${DATASETS[*]}"
uv run python scripts/eval/evaluate.py \
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
