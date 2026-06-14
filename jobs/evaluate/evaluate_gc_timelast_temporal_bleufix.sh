#!/bin/bash
### ============================================================
### DTU HPC — TEMPORAL EVAL (greedy): Phase-1 ARM A, caption-BLEU FAIRNESS re-eval
### Re-evaluates ARM A (sft_gc_timelast_timeaudio_h100, JOBID 28633522) on the
### NEW timestamp-LAST temporal test sets so caption-BLEU is a fair string
### comparison against this model's caption-first output.
###
### WHY: the original eval (28645927) scored ARM A on the timestamp-FIRST sets
### (test_*_temporal_global_caption_anchoroffset.json), whose stored caption is
### verb-spliced ("...and has a relatively low MOS...") with NO "This synthesized
### speech" subject phrase. ARM A trains on timelast targets and emits the FULL
### caption with the subject phrase, so corpus BLEU penalised it for emitting the
### subject it was trained on. The new timelast sets restore the full caption.
###
### SCOPE: this changes ONLY caption_bleu (and the caption-side response-health
### numbers). t-IoU and MOS-MAE are parsed by value, position-independent, and
### are UNCHANGED from 28645927 (mean t-IoU 0.884/0.896/0.871) — do not re-report
### localization from this run, only the corrected caption BLEU.
###
### Writes to a NEW output dir (gc_timelast_greedy_bleufix) so the original
### t-IoU result set (gc_timelast_greedy) is preserved side-by-side.
###
### Mem check (per-core rusage rule): rusage[mem=48GB] x -n 4 = 192 GB total,
### within the gpuh100 ~720 GB node. OK.
###
### Submit with: bsub < jobs/evaluate/evaluate_gc_timelast_temporal_bleufix.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-gc-timelast-bleufix
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 4:00
#BSUB -o logs/eval_timelast_bleufix_%J.out
#BSUB -e logs/eval_timelast_bleufix_%J.err

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

MODEL_PATH="$PROJECT_DIR/models/sft_gc_timelast_timeaudio_h100"
OUTPUT_DIR="$EXPERIMENT_DIR/results/evaluation/temporal/gc_timelast_greedy_bleufix"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Model    : $MODEL_PATH (ARM A, TimeAudio <a><f>)"
echo "Output   : $OUTPUT_DIR"
echo "Test sets: timestamp-LAST (timelast_anchoroffset) — caption-BLEU fairness"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
  echo "ERROR: no checkpoint index at $MODEL_PATH"
  exit 1
fi

DATASETS=(
  "data/processed/temporal/test_FOR_temporal_global_caption_timelast_anchoroffset.json"
  "data/processed/temporal/test_LIVE_temporal_global_caption_timelast_anchoroffset.json"
  "data/processed/temporal/test_P501_temporal_global_caption_timelast_anchoroffset.json"
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
echo "Temporal eval (ARM A, timelast BLEU-fairness) complete: $(date)"
echo "Compare caption_bleu vs the original 28645927 (~16 on the verb-spliced sets)."
echo "=========================================="
