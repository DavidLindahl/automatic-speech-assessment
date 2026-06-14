#!/bin/bash
### ============================================================
### DTU HPC — TEMPORAL EVAL (greedy): DPO ARM A JITTER, caption-BLEU FAIRNESS re-eval
### Re-evaluates the jitter DPO arm (dpo_temporal_armA_jitter, JOBID 28647830)
### on the NEW timestamp-LAST temporal test sets so caption-BLEU is a fair string
### comparison against this model's caption-first output.
###
### WHY: like ARM A, the jitter arm emits caption-FIRST with the full "This
### synthesized speech" subject phrase, but the original jitter evals (28649664
### at 300 tok, 28649679 at 600 tok) scored it on the timestamp-FIRST sets whose
### stored caption is verb-spliced (no subject phrase). corpus BLEU therefore
### penalised it for the subject it was trained on. The new timelast sets restore
### the full caption.
###
### *** 600 TOKENS, NOT 300 ***  jitter DPO inflated captions ~3.2x; at 300 tokens
### the trailing <a><f> clause ran off the end on ~70% of clips (only 54/179
### completed the pair in 28649664). 600 tokens lets the clause finish. This
### matches the canonical jitter eval 28649679, which is the t-IoU of record.
###
### SCOPE: this changes ONLY caption_bleu (and caption-side response-health).
### t-IoU and MOS-MAE are parsed by value, position-independent, and come from
### the 600-token run 28649679 — do not re-report localization from this run,
### only the corrected caption BLEU.
###
### Writes to a NEW output dir (..._greedy_600tok_bleufix) so the existing t-IoU
### result sets are preserved side-by-side.
###
### Mem check (per-core rusage rule): rusage[mem=48GB] x -n 4 = 192 GB total,
### within the gpuh100 ~720 GB node. OK.
###
### Submit with: bsub < jobs/evaluate/evaluate_dpo_armA_jitter_temporal_bleufix.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-dpo-jitter-bleufix
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 4:00
#BSUB -o logs/eval_dpo_jitter_bleufix_%J.out
#BSUB -e logs/eval_dpo_jitter_bleufix_%J.err

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

MODEL_PATH="$PROJECT_DIR/models/dpo_temporal_armA_jitter"
OUTPUT_DIR="$EXPERIMENT_DIR/results/evaluation/temporal/dpo_temporal_armA_jitter_greedy_600tok_bleufix"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Model    : $MODEL_PATH (DPO jitter arm, TimeAudio <a><f>)"
echo "Output   : $OUTPUT_DIR"
echo "Test sets: timestamp-LAST (timelast_anchoroffset) — caption-BLEU fairness"
echo "Tokens   : 600 (avoids the 300-token caption-inflation truncation)"
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
  --max-new-tokens 600 \
  --use-query-prompt

echo "=========================================="
echo "Temporal eval (DPO jitter, timelast BLEU-fairness, 600 tok) complete: $(date)"
echo "Compare caption_bleu vs the 600-token run 28649679 on the verb-spliced sets."
echo "=========================================="
