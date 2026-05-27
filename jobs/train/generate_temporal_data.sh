#!/bin/sh
### ============================================================
### DTU HPC LSF job script — Generate Max NISQA Temporal Mixes
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

echo "Eligible rows (MOS <= 3.0, active tags): $TOTAL_MIX_FILES"
echo "Output dir: $OUTPUT_DIR"

uv run python scripts/data/generate_nisqa_sim_lowmos_active.py \
  --total-mix-files "$TOTAL_MIX_FILES" \
  --mos-max-threshold 3.0 \
  --output-dir "$OUTPUT_DIR" \
  --no-allow-source-reuse \
  --overwrite

uv run python notebooks/build_temporal_inspector_site.py \
  --manifest-path "$OUTPUT_DIR/manifest.csv"

echo "=========================================="
echo "Finished: $(date)"
echo "Manifest: $OUTPUT_DIR/manifest.csv"
echo "Inspector: $OUTPUT_DIR/inspector/index.html"
echo "=========================================="
