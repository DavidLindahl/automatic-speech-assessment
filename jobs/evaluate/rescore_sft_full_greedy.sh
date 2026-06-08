#!/bin/bash
### ============================================================
### DTU HPC — RE-SCORE the full-data SFT greedy eval with ROUGE + BERTScore
###
### The thesis MOS table (tab:full-sft-mos) prints sft_full_paper_h100 greedy
### with MAE/MSE/BLEU only. To put a 5-metric zero-shot baseline row next to it
### (MAE, MSE, BLEU, BERTScore, ROUGE), the SFT row needs the same two caption
### metrics, computed from the SAME stored run so all five numbers are internally
### consistent. `rescore` recomputes BLEU/ROUGE/BERTScore from the stored
### predicted_response, no model inference. roberta-large is already cached.
### --in-place merges the metrics back into each *_results.json.
###
### BERTScore settings (roberta-large, rescale_with_baseline=False) are the
### compute_caption_metrics defaults, identical to the zero-shot eval, so the
### columns are comparable.
###
### Mem check (per-core rusage rule): rusage[mem=32GB] x -n 4 = 128 GB total,
### well within the gpuh100 ~720 GB node. OK.
###
### Submit with: bsub < jobs/evaluate/rescore_sft_full_greedy.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J rescore-sft-full-greedy
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 1:00
#BSUB -o logs/rescore_sft_full_greedy_%J.out
#BSUB -e logs/rescore_sft_full_greedy_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export HF_HOME="$EXPERIMENT_DIR/.cache/huggingface"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    export HF_TOKEN
fi

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

RESDIR="results/evaluation/sft/sft_full_paper_h100_eval_greedy"

uv run python scripts/eval/evaluate.py rescore \
    --results-path "$RESDIR/test_FOR_results.json" \
    --results-path "$RESDIR/test_LIVE_results.json" \
    --results-path "$RESDIR/test_P501_results.json" \
    --results-path "$RESDIR/test_nisqa_indomain_results.json" \
    --in-place

echo "=========================================="
echo "Re-score complete: $(date)"
echo "=========================================="
