#!/bin/bash
### ============================================================
### DTU HPC — ZERO-SHOT TEMPORAL BASELINE SMOKE TEST
### (untrained Qwen2-Audio on temporal localization)
###
### First-contact check before the full temporal baseline eval. Runs the
### off-the-shelf Qwen/Qwen2-Audio-7B-Instruct on just 10 samples each of TWO
### temporal test sets, so we can eyeball the raw model output and, critically,
### the interval-parse behaviour before spending a full 3-dataset run.
###
### This is the temporal counterpart to evaluate_zeroshot_baseline_smoke.sh and
### inherits its hard lesson: a free-text baseline can be parsed into a BOGUS
### interval. The MOS smoke caught extract_mos grabbing the "5" from "3 out of
### 5"; the temporal analogue is the `plain` fallback grabbing any two numbers
### ("rate this 3 / 5, the 6 second clip" -> a fake (3.0, 6.0) interval with a
### non-zero t-IoU). We suppress `plain` under --zero-shot in code, so the audit
### here is to CONFIRM that suppression held and the parse path is honest.
###
### What to check in logs/eval_zeroshot_temporal_smoke_*.out and the JSON:
###   1. "Pred interval sources: {...}" line per dataset. Under --zero-shot it
###      must contain NO "plain" key (the fallback is suppressed). Expect only
###      "range" (honest hits) and "none" (no localizable range emitted).
###   2. predicted_response is coherent text that actually tries to localize,
###      not echo/empty/degenerate output (proves the ChatML prompt is right).
###   3. For a handful of "range" hits, the stated seconds in predicted_response
###      match pred_start/pred_end (the regex picked up the real range, not a
###      stray number).
###   4. A LOW parse rate here is fine and expected: an untrained baseline that
###      cannot emit a clean seconds range is the honest "before" floor, not a
###      bug. A SUSPICIOUSLY HIGH t-IoU is the thing to distrust.
###   5. Caption did not run out of the 300-token budget before the range landed
###      (the instructed baseline is verbose; bump --max-new-tokens if it does).
###
### Only after this looks right, submit evaluate_zeroshot_temporal_baseline.sh.
###
### Mem check (per-core rusage rule): rusage[mem=32GB] x -n 4 = 128 GB total,
### well within the gpuh100 ~720 GB node. OK.
###
### Submit with: bsub < jobs/evaluate/evaluate_zeroshot_temporal_smoke.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-zs-temporal-smoke
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 1:00
#BSUB -o logs/eval_zeroshot_temporal_smoke_%J.out
#BSUB -e logs/eval_zeroshot_temporal_smoke_%J.err

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

# HF auth: Qwen2-Audio-7B-Instruct is downloaded fresh on first load.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
fi

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "GPUs     : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

DATASETS=(
  "data/processed/temporal/test_FOR_temporal_global_caption.json"
  "data/processed/temporal/test_P501_temporal_global_caption.json"
)

for ds in "${DATASETS[@]}"; do
  if [ ! -f "$ds" ]; then
    echo "ERROR: missing dataset $ds"
    exit 1
  fi
done

# Two temporal test sets, 10 samples each, greedy, 300 tokens. --zero-shot
# selects the Instruct baseline, the ChatML prompt, and the plain-fallback
# suppression. The GT interval comes from mix_deg_segments, identical across the
# plain and anchoroffset variants, so this one run is a valid t-IoU baseline
# against both trained temporal models.
uv run python scripts/eval/evaluate_temporal.py \
  --zero-shot \
  --output-dir "$EXPERIMENT_DIR/results/evaluation/temporal/zeroshot_instruct_smoke" \
  --dataset-path "${DATASETS[0]}" \
  --dataset-path "${DATASETS[1]}" \
  --data-root data \
  --max-samples 10 \
  --batch-size 5 \
  --greedy \
  --max-new-tokens 300

echo "=========================================="
echo "Zero-shot TEMPORAL SMOKE complete: $(date)"
echo "Now eyeball 'Pred interval sources' (NO 'plain' key), predicted_response,"
echo "and pred_start/pred_end in the results JSON before submitting"
echo "evaluate_zeroshot_temporal_baseline.sh."
echo "=========================================="
