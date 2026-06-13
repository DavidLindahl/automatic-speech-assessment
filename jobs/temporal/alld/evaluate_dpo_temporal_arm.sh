#!/bin/bash
### ============================================================
### DTU HPC — TEMPORAL EVAL (greedy) for a DPO-aligned ARM A model.
### Parameterized: pass the local model dir name via DPO_MODEL_NAME.
###
###   DPO_MODEL_NAME=dpo_temporal_armA_jitter \
###     bsub < jobs/temporal/alld/evaluate_dpo_temporal_arm.sh
###
### Scores the chosen arm on the filtered FOR/LIVE/P501 temporal test sets in the
### ANCHOROFFSET <a><f> format (MOS<=3, global-caption targets) -> temporal IoU +
### start/end error, greedy. Baseline to beat: ARM A SFT (no DPO) t-IoU
### 0.884/0.896/0.871 on FOR/LIVE/P501.
###
### The arm carries the TimeAudio mechanism (it is a DPO continuation of ARM A);
### load_model auto-detects the subclass from the checkpoint config
### (use_abs_time_embedding/use_time_tokens), so the absolute-time embedding is
### active at inference. No extra flag needed.
###
### Mem check (per-core rusage rule): rusage[mem=48GB] x -n 4 = 192 GB total,
### within the gpuh100 ~720 GB node. OK.
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-dpo-temporal-arm
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 4:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/eval_dpo_temporal_arm_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/eval_dpo_temporal_arm_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

# Which DPO arm to evaluate. Required.
DPO_MODEL_NAME="${DPO_MODEL_NAME:?set DPO_MODEL_NAME to the local model dir name, e.g. dpo_temporal_armA_jitter}"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/results/evaluation"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONPATH=src
export HF_HOME="$EXPERIMENT_DIR/.cache/huggingface"

MODEL_PATH="$PROJECT_DIR/models/$DPO_MODEL_NAME"
OUTPUT_DIR="$EXPERIMENT_DIR/results/evaluation/temporal/${DPO_MODEL_NAME}_greedy"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Model    : $MODEL_PATH (DPO arm, TimeAudio <a><f>)"
echo "Output   : $OUTPUT_DIR"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
  echo "ERROR: no checkpoint index at $MODEL_PATH"
  echo "If the local copy was deleted to free quota, restore from Hub first:"
  echo "  hf download Leng2beat/${DPO_MODEL_NAME//_/-} --local-dir $MODEL_PATH"
  exit 1
fi

DATASETS=(
  "data/processed/temporal/test_FOR_temporal_global_caption_anchoroffset.json"
  "data/processed/temporal/test_LIVE_temporal_global_caption_anchoroffset.json"
  "data/processed/temporal/test_P501_temporal_global_caption_anchoroffset.json"
)

for ds in "${DATASETS[@]}"; do
  if [ ! -f "$ds" ]; then
    echo "ERROR: missing dataset $ds"
    exit 1
  fi
done

uv run python scripts/eval/evaluate_temporal.py \
  --model-path "$MODEL_PATH" \
  --dataset-path "${DATASETS[0]}" \
  --dataset-path "${DATASETS[1]}" \
  --dataset-path "${DATASETS[2]}" \
  --data-root data \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 4 \
  --greedy \
  --temperature 0.0 \
  --max-new-tokens 300 \
  --use-query-prompt

echo "=========================================="
echo "Temporal eval ($DPO_MODEL_NAME, greedy) complete: $(date)"
echo "=========================================="
