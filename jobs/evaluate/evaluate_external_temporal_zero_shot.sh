#!/bin/bash
### ============================================================
### DTU HPC — External temporal zero-shot evaluation
### Submit with: bsub < jobs/evaluate/evaluate_external_temporal_zero_shot.sh
###
### Default backend: TimeAudio.
### Required for TimeAudio:
###   TIMEAUDIO_DIR=/work3/s234817/TimeAudio
###   TIMEAUDIO_PYTHON=$TIMEAUDIO_DIR/.venv/bin/python
###   checkpoints under $TIMEAUDIO_DIR/pretrained_model/
###
### SALMONN support starts from the same prepared JSON files and scorer.
### Add only a backend runner once the exact SALMONN inference CLI is chosen.
### ============================================================

#BSUB -q gpuh100
#BSUB -J eval-external-temporal-zero-shot
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 12:00
#BSUB -o logs/evaluate_external_temporal_zero_shot_%J.out
#BSUB -e logs/evaluate_external_temporal_zero_shot_%J.err

source "$(dirname "$0")/../_lib/preamble.sh"

BACKEND="${BACKEND:-timeaudio}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
QUESTION="${QUESTION:-Please describe and evaluate the synthetic speech, and find timestamps for the degradation.}"

RUN_NAME="${RUN_NAME:-${BACKEND}_zero_shot}"
OUTPUT_DIR="results/evaluation/temporal/${RUN_NAME}"
INPUT_DIR="$OUTPUT_DIR/inputs"
RAW_PRED_DIR="$OUTPUT_DIR/raw_predictions"
SCORED_DIR="$OUTPUT_DIR/scored"

DATASETS=(
  "data/processed/temporal/test_FOR_temporal.json"
  "data/processed/temporal/test_LIVE_temporal.json"
  "data/processed/temporal/test_P501_temporal.json"
)

mkdir -p "$INPUT_DIR" "$RAW_PRED_DIR" "$SCORED_DIR"

echo "Backend  : $BACKEND"
echo "Output   : $OUTPUT_DIR"
echo "Question : $QUESTION"

prepare_dataset() {
  local dataset_path="$1"
  uv run python scripts/eval/external_temporal.py prepare \
    --dataset-path "$dataset_path" \
    --output-dir "$INPUT_DIR" \
    --data-root data \
    --model-format "$BACKEND" \
    --question "$QUESTION"
}

score_predictions() {
  local dataset_path="$1"
  local prediction_path="$2"
  local dataset_stem
  dataset_stem="$(basename "$dataset_path" .json)"
  uv run python scripts/eval/external_temporal.py score \
    --dataset-path "$dataset_path" \
    --prediction-path "$prediction_path" \
    --output-json "$SCORED_DIR/${dataset_stem}_results.json" \
    --output-csv "$SCORED_DIR/${dataset_stem}_results.csv"
}

run_timeaudio() {
  local dataset_path="$1"
  local dataset_stem="$2"
  local input_json="$3"
  local prediction_json="$4"

  TIMEAUDIO_DIR="${TIMEAUDIO_DIR:-/work3/s234817/TimeAudio}"
  TIMEAUDIO_PYTHON="${TIMEAUDIO_PYTHON:-$TIMEAUDIO_DIR/.venv/bin/python}"
  TIMEAUDIO_CFG="${TIMEAUDIO_CFG:-$TIMEAUDIO_DIR/configs/infer_config.yaml}"
  TIMEAUDIO_PRETRAINED="${TIMEAUDIO_PRETRAINED:-$TIMEAUDIO_DIR/pretrained_model}"
  TIMEAUDIO_VICUNA_PATH="${TIMEAUDIO_VICUNA_PATH:-$TIMEAUDIO_PRETRAINED/vicuna-7b-v1.5}"
  TIMEAUDIO_WHISPER_PATH="${TIMEAUDIO_WHISPER_PATH:-$TIMEAUDIO_PRETRAINED/whisper-large-v2}"
  TIMEAUDIO_BEATS_PATH="${TIMEAUDIO_BEATS_PATH:-$TIMEAUDIO_PRETRAINED/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt}"
  TIMEAUDIO_CKPT="${TIMEAUDIO_CKPT:-$TIMEAUDIO_PRETRAINED/timeaudio.pth}"

  if [ ! -d "$TIMEAUDIO_DIR" ]; then
    echo "Missing TIMEAUDIO_DIR: $TIMEAUDIO_DIR"
    echo "Clone https://github.com/lysanderism/TimeAudio there, then rerun."
    exit 1
  fi
  if [ ! -x "$TIMEAUDIO_PYTHON" ]; then
    echo "Missing executable TIMEAUDIO_PYTHON: $TIMEAUDIO_PYTHON"
    echo "Create a TimeAudio environment, or set TIMEAUDIO_PYTHON explicitly."
    exit 1
  fi
  if [ ! -f "$TIMEAUDIO_CKPT" ]; then
    echo "Missing TimeAudio checkpoint: $TIMEAUDIO_CKPT"
    exit 1
  fi

  echo "Running TimeAudio on $dataset_path"
  (
    cd "$TIMEAUDIO_DIR"
    "$TIMEAUDIO_PYTHON" -m torch.distributed.run \
      --nproc_per_node "$NPROC_PER_NODE" \
      inference.py \
      --cfg-path "$TIMEAUDIO_CFG" \
      --options \
      datasets.test_ann_path="$PROJECT_DIR/$input_json" \
      datasets.test_save_path="$PROJECT_DIR/$prediction_json" \
      datasets.whisper_path="$TIMEAUDIO_WHISPER_PATH" \
      datasets.format_tokens=v3 \
      run.output_dir="$PROJECT_DIR/$OUTPUT_DIR/timeaudio_logs/$dataset_stem" \
      run.world_size="$NPROC_PER_NODE" \
      run.batch_size_eval="$BATCH_SIZE" \
      run.num_workers="$NUM_WORKERS" \
      run.use_distributed=true \
      model.llama_path="$TIMEAUDIO_VICUNA_PATH" \
      model.whisper_path="$TIMEAUDIO_WHISPER_PATH" \
      model.beats_path="$TIMEAUDIO_BEATS_PATH" \
      model.ckpt="$TIMEAUDIO_CKPT"
  )
}

case "$BACKEND" in
  timeaudio|salmonn) ;;
  *)
    echo "Unsupported BACKEND: $BACKEND"
    echo "Use BACKEND=timeaudio or BACKEND=salmonn."
    exit 1
    ;;
esac

for dataset_path in "${DATASETS[@]}"; do
  prepare_dataset "$dataset_path"
done

for dataset_path in "${DATASETS[@]}"; do
  dataset_stem="$(basename "$dataset_path" .json)"
  input_json="$INPUT_DIR/${dataset_stem}_${BACKEND}.json"
  prediction_json="$RAW_PRED_DIR/${dataset_stem}_${BACKEND}_predictions.json"

  case "$BACKEND" in
    timeaudio)
      run_timeaudio "$dataset_path" "$dataset_stem" "$input_json" "$prediction_json"
      ;;
    salmonn)
      echo "Prepared SALMONN-compatible input: $input_json"
      echo "SALMONN runner is intentionally not guessed yet. Add the exact CLI here."
      exit 2
      ;;
  esac

  score_predictions "$dataset_path" "$prediction_json"
done

echo "External temporal zero-shot evaluation complete: $(date)"
