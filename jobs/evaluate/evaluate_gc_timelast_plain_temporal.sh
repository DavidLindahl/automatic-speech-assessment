#!/bin/bash
### ============================================================
### DTU HPC — TEMPORAL EVAL (greedy): PHASE-1 ARM A basic twin
### (caption-last, vanilla CE, PLAIN <|s|> timestamps, no time mechanism).
###
### Evaluates sft_gc_timelast_plain_h100 on the FOR/LIVE/P501 temporal test sets
### in the PLAIN <|s|> timestamp-LAST format. Reports the FULL panel: t-IoU +
### auto audio-blind baselines + unique intervals + response health (unique
### captions, caption BLEU, MOS MAE).
###
### This is the fair basic-setup counterpart of evaluate_gc_timelast_temporal.sh
### (ARM A, TimeAudio). Same answer order (timestamp last), same decoding
### (greedy), same 300-token budget, same full panel — the ONLY axis that varies
### is the timestamp mechanism (this one: none / plain <|s|>; ARM A: TimeAudio
### <a><f> + abs-time embedding). Compare the two runs directly for the
### basic-vs-TimeAudio temporal comparison at matched order.
###
### Matching-format eval: this model emits <|seconds|> timestamps, so it is
### scored on the plain <|s|> timelast test variants. GT comes from
### mix_deg_segments (format-agnostic), but the query/format must match what the
### model was trained on. load_model needs no time-mechanism flag here — the
### checkpoint config carries no TimeAudio subclass.
###
### Mem check (per-core rusage rule): rusage[mem=48GB] x -n 4 = 192 GB total,
### within the gpuh100 ~720 GB node. OK.
###
### Submit with: bsub < jobs/evaluate/evaluate_gc_timelast_plain_temporal.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-gc-timelast-plain-temporal
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 4:00
#BSUB -o logs/eval_timelast_plain_temporal_%J.out
#BSUB -e logs/eval_timelast_plain_temporal_%J.err

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

MODEL_PATH="$PROJECT_DIR/models/sft_gc_timelast_plain_h100"
OUTPUT_DIR="$EXPERIMENT_DIR/results/evaluation/temporal/gc_timelast_plain_greedy"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Model    : $MODEL_PATH (ARM A basic twin, plain <|s|>)"
echo "Output   : $OUTPUT_DIR"
echo "Test sets: plain <|s|> timestamp-LAST (timelast)"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
  echo "ERROR: no checkpoint index at $MODEL_PATH"
  exit 1
fi

DATASETS=(
  "data/processed/temporal/test_FOR_temporal_global_caption_timelast.json"
  "data/processed/temporal/test_LIVE_temporal_global_caption_timelast.json"
  "data/processed/temporal/test_P501_temporal_global_caption_timelast.json"
)

for ds in "${DATASETS[@]}"; do
  if [ ! -f "$ds" ]; then
    echo "ERROR: missing dataset $ds"
    echo "Build first: bsub < jobs/evaluate/build_timelast_plain_test_sets.sh"
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
echo "Temporal eval (gc-timelast-plain, greedy) complete: $(date)"
echo "Compare against ARM A (evaluate_gc_timelast_temporal.sh, TimeAudio) for"
echo "the basic-vs-TimeAudio temporal comparison at matched answer order."
echo "=========================================="
