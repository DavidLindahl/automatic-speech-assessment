#!/bin/bash
### ============================================================
### DTU HPC — ZERO-SHOT BASELINE SMOKE TEST (untrained Qwen2-Audio)
###
### First-contact check before the full baseline eval. Runs the off-the-shelf
### Qwen/Qwen2-Audio-7B-Instruct on just THREE samples of ONE dataset, so we
### can eyeball the raw model output before spending a full 4-dataset run.
###
### What to check in logs/eval_zeroshot_smoke_*.out and the results JSON:
###   1. predicted_response is coherent text, not echo/empty/degenerate output
###      (proves the chat template prompts the Instruct model correctly).
###   2. The model actually ends with a stated MOS score.
###   3. predicted_mos == the number the model stated (not prompt residue like a
###      dimension index or an echoed "1 to 5" caught by extract_mos's fallback).
###   4. The caption did NOT run out of the 150-token budget before the MOS
###      (the instructed baseline is more verbose than the trained models; bump
###      --max-new-tokens here and in evaluate_zeroshot_baseline.sh if it does).
###
### Only after this looks right, submit jobs/evaluate/evaluate_zeroshot_baseline.sh.
###
### Mem check (per-core rusage rule): rusage[mem=32GB] x -n 4 = 128 GB total,
### well within the gpuh100 ~720 GB node. OK.
###
### Submit with: bsub < jobs/evaluate/evaluate_zeroshot_baseline_smoke.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-zeroshot-smoke
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 1:00
#BSUB -o logs/eval_zeroshot_smoke_%J.out
#BSUB -e logs/eval_zeroshot_smoke_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/results/evaluation"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
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

# Two datasets, 10 samples each, spanning the MOS range (the first 10 of FOR
# and P501 both run ~2.0 to ~4.9, so the parser is exercised on low ratings
# where the old "X out of 5" denominator-grab would have caused ~3-point
# errors). max-new-tokens raised to 300: the 150-token first smoke truncated
# captions before the score landed.
uv run python scripts/eval/evaluate.py eval-mos \
    --zero-shot \
    --output-dir "$EXPERIMENT_DIR/results/evaluation/zeroshot/qwen2audio_instruct_corrected_prompt_smoke" \
    --dataset-path "data/processed/eval/test_FOR.json" \
    --dataset-path "data/processed/eval/test_P501.json" \
    --max-samples 10 \
    --batch-size 5 \
    --greedy \
    --max-new-tokens 300

echo "=========================================="
echo "Zero-shot baseline SMOKE complete: $(date)"
echo "Now eyeball the predicted_response / predicted_mos in the results JSON"
echo "before submitting evaluate_zeroshot_baseline.sh."
echo "=========================================="
