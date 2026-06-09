#!/bin/bash
### ============================================================
### DTU HPC — ZERO-SHOT TEMPORAL BASELINE (FULL)
### Untrained Qwen/Qwen2-Audio-7B-Instruct on the temporal localization task.
###
### The defensible "before fine-tuning" t-IoU floor for the temporal results
### tables: off-the-shelf model, no training, ChatML chat-template prompt
### (non-leaking, no examples), greedy, 300 tokens, on all three temporal test
### sets (FOR, LIVE, P501). This is the temporal counterpart to the MOS
### zero-shot baseline (evaluate_zeroshot_baseline.sh) and reproduces the source
### paper's finding that off-the-shelf audio LLMs cannot do this task without
### fine-tuning, here on the "when is the degradation" sub-task.
###
### GATED BY THE SMOKE (evaluate_zeroshot_temporal_smoke.sh): do not submit this
### until the smoke confirms the parse path is honest, the "Pred interval
### sources" tally has NO "plain" key (the manufactured-interval fallback is
### suppressed under --zero-shot), and a low parse rate / low t-IoU is the
### result rather than a suspiciously competent one.
###
### The GT interval comes from mix_deg_segments, identical across the plain and
### anchoroffset global-caption test variants, so this single run is a valid
### t-IoU baseline against BOTH trained temporal models (28612484 plain-token
### SFT and 28612486 TimeAudio-method SFT).
###
### Mem check (per-core rusage rule): rusage[mem=48GB] x -n 4 = 192 GB total,
### well within the gpuh100 ~720 GB node. OK.
###
### Submit with: bsub < jobs/evaluate/evaluate_zeroshot_temporal_baseline.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-zs-temporal
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 4:00
#BSUB -o logs/eval_zeroshot_temporal_%J.out
#BSUB -e logs/eval_zeroshot_temporal_%J.err

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
echo "Model    : Qwen/Qwen2-Audio-7B-Instruct (off-the-shelf, untrained)"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

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
  --zero-shot \
  --output-dir "$EXPERIMENT_DIR/results/evaluation/temporal/zeroshot_instruct_baseline" \
  --dataset-path "${DATASETS[0]}" \
  --dataset-path "${DATASETS[1]}" \
  --dataset-path "${DATASETS[2]}" \
  --data-root data \
  --batch-size 8 \
  --greedy \
  --max-new-tokens 300

echo "=========================================="
echo "Zero-shot TEMPORAL baseline (full) complete: $(date)"
echo "=========================================="
