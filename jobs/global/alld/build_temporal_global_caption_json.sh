#!/bin/sh
### ============================================================
### DTU HPC LSF job script — Build temporal training JSON in the
### "global-caption" format: the global MOS caption with a leading
### temporal clause. Builds BOTH timestamp variants from the SAME
### max_mos3 mixes:
###   1. global-caption-localization  -> free-text <|seconds|> tokens
###   2. global-caption-anchoroffset  -> discrete <aN><fK> tokens
### Reuses the existing nisqa_sim_mix_lowmos_active_max_mos3 mixes
### (no audio regen). Submit with:
###   bsub < jobs/train/build_temporal_global_caption_json.sh
### ============================================================

#BSUB -q hpc
#BSUB -J build-temporal-global-caption
#BSUB -n 2
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 04:00
#BSUB -o logs/build_temporal_global_caption_%J.out
#BSUB -e logs/build_temporal_global_caption_%J.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/work3/s234817/automatic-speech-assessment}"
cd "$PROJECT_DIR"

mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

# Same mixes that built train_nisqa_temporal_mix_max_mos3_localized.json.
MIXES_DIR="data/processed/temporal/nisqa_sim_mix_lowmos_active_max_mos3"
MANIFEST_PATH="$MIXES_DIR/manifest.csv"
CAPTION_JSONL="data/processed/sft/train_nisqa_llama_10k.json"

OUT_LOCALIZATION="data/processed/temporal/train_nisqa_temporal_global_caption.json"
OUT_ANCHOROFFSET="data/processed/temporal/train_nisqa_temporal_global_caption_anchoroffset.json"

# New temporal query (typo-fixed). Kept identical across both variants and the
# test sets so train and eval prompts match.
QUERY="Please describe and evaluate the synthetic speech, and identify when the degradation occurs.<audio>"

echo "=========================================="
echo "Job ID   : ${LSB_JOBID:-local}"
echo "Host     : $(hostname)"
echo "Started  : $(date)"
echo "Mixes    : $MIXES_DIR"
echo "Manifest : $MANIFEST_PATH"
echo "Query    : $QUERY"
echo "=========================================="

if [ ! -f "$MANIFEST_PATH" ]; then
  echo "Missing manifest: $MANIFEST_PATH"
  echo "Run first: bsub < jobs/train/generate_nisqa_temporal_max.sh"
  exit 1
fi

# Expected record count = the existing localized baseline (built from the same
# manifest+mixes). Used to fail loudly if a path mismatch silently drops rows.
EXPECTED_JSON="data/processed/temporal/train_nisqa_temporal_mix_max_mos3_localized.json"
EXPECTED_COUNT=""
if [ -f "$EXPECTED_JSON" ]; then
  EXPECTED_COUNT="$(wc -l < "$EXPECTED_JSON" | tr -d ' ')"
  echo "Expected record count (from $EXPECTED_JSON): $EXPECTED_COUNT"
fi

build_variant() {
  style="$1"
  out_path="$2"
  echo "------------------------------------------"
  echo "Building label_style=$style -> $out_path"
  uv run python scripts/data/build_nisqa_temporal_json.py \
    --manifest-path "$MANIFEST_PATH" \
    --mixes-dir "$MIXES_DIR" \
    --caption-jsonl "$CAPTION_JSONL" \
    --output-jsonl "$out_path" \
    --label-style "$style" \
    --query "$QUERY"

  got="$(wc -l < "$out_path" | tr -d ' ')"
  echo "Wrote $got records to $out_path"
  if [ -n "$EXPECTED_COUNT" ] && [ "$got" != "$EXPECTED_COUNT" ]; then
    echo "ERROR: record count $got != expected $EXPECTED_COUNT for $out_path"
    echo "A mismatch usually means wrong manifest/mixes dir or missing mix files."
    exit 1
  fi
}

build_variant "global-caption-localization" "$OUT_LOCALIZATION"
build_variant "global-caption-anchoroffset" "$OUT_ANCHOROFFSET"

echo "=========================================="
echo "First record (localization):"
head -n 1 "$OUT_LOCALIZATION"
echo ""
echo "First record (anchoroffset):"
head -n 1 "$OUT_ANCHOROFFSET"
echo "=========================================="
echo "Done: $(date)"
