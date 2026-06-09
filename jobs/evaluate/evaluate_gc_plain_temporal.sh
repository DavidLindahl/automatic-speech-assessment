#!/bin/bash
### ============================================================
### DTU HPC — TEMPORAL EVAL (greedy): global-caption PLAIN SFT model
### Evaluates the Setup-1 model (28615748, plain SFT, free-text <|s|>
### timestamps, no time mechanism) on the filtered FOR/LIVE/P501 temporal
### test sets in the PLAIN <|s|> format (MOS<=3, global-caption targets).
### Temporal IoU + start/end error, greedy decoding.
###
### Matching-format eval: this model emits <|seconds|> timestamps, so it is
### scored on the <|s|> test variants. (GT comes from mix_deg_segments, so the
### metric is format-agnostic; the query/format still must match what the model
### was trained on.)
###
### Mem check (per-core rusage rule): rusage[mem=48GB] x -n 4 = 192 GB total,
### within the gpuh100 ~720 GB node. OK.
###
### Submit with: bsub < jobs/evaluate/evaluate_gc_plain_temporal.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-gc-plain-temporal
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 4:00
#BSUB -o logs/eval_gc_plain_temporal_%J.out
#BSUB -e logs/eval_gc_plain_temporal_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/results/evaluation"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONPATH=src
export HF_HOME="$EXPERIMENT_DIR/.cache/huggingface"

MODEL_PATH="$PROJECT_DIR/models/sft_temporal_gc_plain_h100"
OUTPUT_DIR="$EXPERIMENT_DIR/results/evaluation/temporal/gc_plain_greedy"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Model    : $MODEL_PATH (Setup 1, plain <|s|>)"
echo "Output   : $OUTPUT_DIR"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
  echo "ERROR: no checkpoint index at $MODEL_PATH"
  exit 1
fi

DATASETS=(
  "data/processed/temporal/test_FOR_temporal_global_caption.json"
  "data/processed/temporal/test_LIVE_temporal_global_caption.json"
  "data/processed/temporal/test_P501_temporal_global_caption.json"
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
echo "Temporal eval (gc-plain, greedy) complete: $(date)"
echo "=========================================="
