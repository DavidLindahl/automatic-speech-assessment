#!/bin/sh
### ============================================================
### DTU HPC LSF job script — Generate Max NISQA Temporal Mixes + JSONL
### Submit with: bsub < jobs/train/generate_nisqa_temporal_max.sh
### ============================================================

### -- Queue: CPU queue (audio processing only) --
#BSUB -q hpc

### -- Job name --
#BSUB -J gen-nisqa-temporal-max

### -- CPU cores --
#BSUB -n 8

### -- System memory --
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB

### -- Walltime --
#BSUB -W 24:00

### -- Output / error files --
#BSUB -o logs/gen_nisqa_temporal_max_%J.out
#BSUB -e logs/gen_nisqa_temporal_max_%J.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/work3/s234817/automatic-speech-assessment}"
cd "$PROJECT_DIR"

mkdir -p logs

source .venv/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

echo "=========================================="
echo "Job ID   : ${LSB_JOBID:-local}"
echo "Host     : $(hostname)"
echo "Started  : $(date)"
echo "Project  : $PROJECT_DIR"
echo "=========================================="

TOTAL_MIX_FILES="$(uv run python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

data_root = Path("data/raw/NISQA_Corpus")
sim_split = "NISQA_TRAIN_SIM"
csv_path = data_root / sim_split / f"{sim_split}_file.csv"
mos_max = 3.0
degradation_columns = [
    "filter",
    "timeclipping",
    "wbgn",
    "p50mnru",
    "bgn",
    "clipping",
    "arb_filter",
    "codec1",
    "codec2",
    "codec3",
    "plcMode1",
    "plcMode2",
    "plcMode3",
]

def is_active(value: object) -> bool:
    if pd.isna(value):
        return False
    token = str(value).strip()
    return token not in {"", "-", "nan", "None"}

df = pd.read_csv(csv_path)
df["ref_path"] = df["filepath_ref"].apply(lambda p: data_root / p)
df["deg_path"] = df["filepath_deg"].apply(lambda p: data_root / p)
df = df[df["ref_path"].apply(Path.exists) & df["deg_path"].apply(Path.exists)].copy()
df = df[df["mos"].notna() & (df["mos"] <= mos_max)].copy()
df["num_active"] = df.apply(
    lambda row: sum(is_active(row.get(col, np.nan)) for col in degradation_columns),
    axis=1,
)
df = df[df["num_active"] > 0].copy()
print(len(df))
PY
)"

if [ -z "$TOTAL_MIX_FILES" ] || [ "$TOTAL_MIX_FILES" -le 0 ]; then
  echo "Could not determine eligible sample count. Aborting."
  exit 1
fi

OUTPUT_DIR="data/processed/temporal/nisqa_sim_mix_lowmos_active_max_mos3"
TRAIN_JSON="data/processed/temporal/train_nisqa_temporal_mix_max_mos3_localized.json"
JOB_TAG="${LSB_JOBID:-local}"
GEN_LOG="logs/gen_nisqa_temporal_max_${JOB_TAG}.tmp.log"

echo "Eligible rows (MOS <= 3.0, active tags): $TOTAL_MIX_FILES"
echo "Output dir: $OUTPUT_DIR"
echo "Train json: $TRAIN_JSON"

generate_dataset() {
  target_files="$1"
  uv run python scripts/data/generate_nisqa_sim_lowmos_active.py \
    --total-mix-files "$target_files" \
    --mos-max-threshold 3.0 \
    --output-dir "$OUTPUT_DIR" \
    --allow-source-reuse \
    --max-source-passes 25 \
    --max-row-attempts 12 \
    --output-active-fraction-min 0.40 \
    --overwrite
}

if ! generate_dataset "$TOTAL_MIX_FILES" 2>&1 | tee "$GEN_LOG"; then
  FALLBACK_COUNT="$(sed -n 's/.*Generated only \([0-9][0-9]*\) files out of requested.*/\1/p' "$GEN_LOG" | tail -n 1)"
  if [ -z "$FALLBACK_COUNT" ] || [ "$FALLBACK_COUNT" -le 0 ]; then
    echo "Generation failed and no fallback count was parsed."
    exit 1
  fi
  echo "Retrying generation with fallback target: $FALLBACK_COUNT"
  generate_dataset "$FALLBACK_COUNT"
fi

uv run python notebooks/build_temporal_inspector_site.py \
  --manifest-path "$OUTPUT_DIR/manifest.csv"

uv run python scripts/data/build_nisqa_temporal_json.py \
  --manifest-path "$OUTPUT_DIR/manifest.csv" \
  --mixes-dir "$OUTPUT_DIR" \
  --caption-jsonl data/processed/sft/train_nisqa_llama_10k.json \
  --output-jsonl "$TRAIN_JSON"

echo "=========================================="
echo "Finished: $(date)"
echo "Manifest: $OUTPUT_DIR/manifest.csv"
echo "Inspector: $OUTPUT_DIR/inspector/index.html"
echo "Train JSON: $TRAIN_JSON"
echo "=========================================="
