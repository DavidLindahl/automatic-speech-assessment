#!/bin/sh
### ============================================================
### DTU HPC LSF job script — Generate placement-AUGMENTED NISQA temporal mixes.
###
### Same single-segment supervision as generate_nisqa_temporal_max.sh, but each
### source REF/DEG pair is reused to EXHAUSTION: every reuse places the
### degradation in a NEW non-overlapping active-speech region, so one REF yields
### several one-segment training files. Windows are shortened (0.6-2.25s) so more
### fit per clip; a 300-source dry-run measured ~2.3x augmentation at these knobs.
###
### The SFT target shape is UNCHANGED (one <aN><fK> interval per record); only the
### dataset size grows. Output goes to a NEW dir so the existing 5,136-record
### baseline (train_nisqa_temporal_anchoroffset.json) is left intact for the
### augmented-vs-baseline comparison.
###
### NOTE: the existing baseline used 1.0s/0.40 windows; this set uses 0.6s/0.25,
### so the augmented-vs-baseline IoU comparison is confounded by window length.
### Treat that as a named limitation in the results/discussion chapter.
###
### Submit with: bsub < jobs/temporal/data/generate_nisqa_temporal_augmented.sh
### ============================================================

### -- Queue: CPU queue (audio processing only) --
#BSUB -q hpc

### -- Job name --
#BSUB -J gen-nisqa-temporal-aug

### -- CPU cores --
#BSUB -n 8

### -- System memory: 16GB x 8 cores = 128GB total (per-core rusage, fits hpc) --
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB

### -- Walltime --
#BSUB -W 24:00

### -- Output / error files --
#BSUB -o logs/gen_nisqa_temporal_aug_%J.out
#BSUB -e logs/gen_nisqa_temporal_aug_%J.err

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
echo "Mode     : placement-augmented (exhaustion, non-overlapping reuse)"
echo "Knobs    : segment-min-seconds=0.6 segment-max-ratio=0.25"
echo "=========================================="

OUTPUT_DIR="data/processed/temporal/nisqa_sim_mix_lowmos_active_aug_mos3"
TRAIN_JSON="data/processed/temporal/train_nisqa_temporal_anchoroffset_aug.json"

echo "Output dir: $OUTPUT_DIR"
echo "Train json: $TRAIN_JSON"

### Exhaustion mode: --total-mix-files 0 lets the generator keep reusing each
### source until no non-overlapping active window remains. The per-core
### rusage[mem] rule is respected (16GB x 8 = 128GB).
uv run python scripts/data/generate_nisqa_sim_lowmos_active.py \
  --total-mix-files 0 \
  --mos-max-threshold 3.0 \
  --output-dir "$OUTPUT_DIR" \
  --allow-source-reuse \
  --max-source-passes 25 \
  --max-row-attempts 12 \
  --segment-min-seconds 0.6 \
  --segment-max-ratio 0.25 \
  --output-active-fraction-min 0.40 \
  --overwrite

### Build the inspector site for visual QA of the placements.
uv run python scripts/data/build_temporal_inspector_site.py \
  --manifest-path "$OUTPUT_DIR/manifest.csv"

### Build the SFT JSONL in the SAME anchor-offset format as the current
### TimeAudio target, so the augmented set is a drop-in replacement.
uv run python scripts/data/build_nisqa_temporal_json.py \
  --manifest-path "$OUTPUT_DIR/manifest.csv" \
  --mixes-dir "$OUTPUT_DIR" \
  --caption-jsonl data/processed/sft/train_nisqa_llama_10k.json \
  --output-jsonl "$TRAIN_JSON" \
  --label-style anchor-offset-localization

echo "=========================================="
echo "Finished: $(date)"
echo "Manifest: $OUTPUT_DIR/manifest.csv"
echo "Inspector: $OUTPUT_DIR/inspector/index.html"
echo "Train JSON: $TRAIN_JSON"
echo "Record count:"
wc -l "$TRAIN_JSON" || true
echo "=========================================="
