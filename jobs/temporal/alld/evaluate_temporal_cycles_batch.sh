#!/bin/bash
### ============================================================
### DTU HPC — TEMPORAL CYCLES BATCH RE-SCORE (greedy, timestamp-last).
###
### Re-scores the SFT baseline + every temporal DPO cycle arm in ONE job, with
### the EXTENDED eval (scripts/eval/evaluate_temporal.py now emits MOS MAE/MSE +
### caption BLEU/ROUGE-L/BERTScore alongside t-IoU). Fills the whole
### tab:temporal-cycles grid from one run.
###
###   bsub < jobs/temporal/alld/evaluate_temporal_cycles_batch.sh
###
### One arm at a time to respect the /work3 100 GiB quota (~16 GB/model): for
### each arm we download the checkpoint from the Hub, evaluate, then DELETE the
### local copy before the next arm. Only ~16 GB of model weights are on disk at
### any moment on top of the ~66 GB already used.
###
### Scoring is on the timestamp-LAST sets (TEST_VARIANT=timelast_anchoroffset),
### MAX_NEW_TOKENS=600 (DPO can inflate captions; 300 truncated the trailing
### <a><f> clause on ~70% of clips for the jitter arm), greedy. t-IoU is
### position-independent (parsed by value), so it is identical to the timestamp-
### first sets; only the caption metrics need the matched order.
###
### Mem check (per-core rusage rule): rusage[mem=48GB] x -n 4 = 192 GB total,
### within the gpuh100 ~720 GB node. OK.
###
### PRECONDITION (verified before submit): every repo in ARMS below must EXIST on
### the Hub with weights. As of writing only sft-gc-timelast-timeaudio and
### dpo-temporal-armA-ts-sampled exist; cycle-caption (PEND), cycle-mos and full
### are not yet trained. Do NOT submit until all five are on the Hub, or trim
### ARMS to the ones that exist.
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-temporal-cycles-batch
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 12:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/eval_temporal_cycles_batch_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/eval_temporal_cycles_batch_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-600}"
TEST_VARIANT="${TEST_VARIANT:-timelast_anchoroffset}"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/results/evaluation"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONPATH=src
export HF_HOME="$EXPERIMENT_DIR/.cache/huggingface"

# Arms to re-score, as "local_model_dir_name  Hub_repo" pairs. The SFT baseline
# is included (its caption BLEU shifts under the new timestamp-stripped scoring,
# so the whole grid column must be recomputed consistently).
ARMS=(
  "sft_gc_timelast_timeaudio_h100   Leng2beat/sft-gc-timelast-timeaudio"
  "dpo_temporal_armA_ts_sampled     Leng2beat/dpo-temporal-armA-ts-sampled"
  "dpo_temporal_armA_cycle_caption  Leng2beat/dpo-temporal-armA-cycle-caption"
  "dpo_temporal_armA_cycle_mos      Leng2beat/dpo-temporal-armA-cycle-mos"
  "dpo_temporal_armA_full           Leng2beat/dpo-temporal-armA-full"
)

DATASETS=(
  "data/processed/temporal/test_FOR_temporal_global_caption_${TEST_VARIANT}.json"
  "data/processed/temporal/test_LIVE_temporal_global_caption_${TEST_VARIANT}.json"
  "data/processed/temporal/test_P501_temporal_global_caption_${TEST_VARIANT}.json"
)

for ds in "${DATASETS[@]}"; do
  if [ ! -f "$ds" ]; then echo "ERROR: missing dataset $ds"; exit 1; fi
done

echo "=========================================="
echo "Job ID    : $LSB_JOBID"
echo "Host      : $(hostname)"
echo "Branch    : $(git branch --show-current 2>/dev/null || echo unknown)"
echo "Commit    : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "TestVar   : $TEST_VARIANT"
echo "MaxTokens : $MAX_NEW_TOKENS"
echo "Arms      : ${#ARMS[@]}"
echo "Started   : $(date)"
echo "=========================================="
nvidia-smi

for entry in "${ARMS[@]}"; do
  read -r MODEL_DIR HUB_REPO <<<"$entry"
  MODEL_PATH="$PROJECT_DIR/models/$MODEL_DIR"
  OUTPUT_DIR="$EXPERIMENT_DIR/results/evaluation/temporal/${MODEL_DIR}_greedy_timelast_full"

  echo ""
  echo ">>> ===== ARM: $MODEL_DIR  (Hub: $HUB_REPO) ====="
  echo ">>> Output: $OUTPUT_DIR"
  echo ">>> Disk before download (project dir usage):"
  du -sh "$PROJECT_DIR" 2>/dev/null | head -1 || true

  # Download from Hub only if the local checkpoint is not already present.
  if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
    echo ">>> Downloading $HUB_REPO -> $MODEL_PATH"
    hf download "$HUB_REPO" --local-dir "$MODEL_PATH" \
      || { echo "ERROR: download failed for $HUB_REPO"; exit 1; }
  else
    echo ">>> Local checkpoint already present, skipping download."
  fi

  echo ">>> Evaluating $MODEL_DIR (greedy, $MAX_NEW_TOKENS tok, $TEST_VARIANT)"
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
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --use-query-prompt

  # Delete the local checkpoint to free quota before the next arm. The weights
  # stay on the Hub; this directory is a transient download. (The SFT base copy
  # is also removed here; restore from the Hub if a later job needs it.)
  echo ">>> Deleting local checkpoint $MODEL_PATH to free quota"
  rm -rf "$MODEL_PATH"
  echo ">>> Done arm $MODEL_DIR: $(date)"
done

echo "=========================================="
echo "Temporal cycles batch re-score complete: $(date)"
echo "Results under results/evaluation/temporal/*_greedy_timelast_full/"
echo "=========================================="
