#!/bin/bash
### ============================================================
### DTU HPC — FRAME PROBE (Phase-0 gate for the temporal loss design)
### Decides whether the temporal SFT collapse (28618993/28618994: both arms
### below the audio-blind baselines) is an objective problem or a
### representation problem. Runs the frozen audio tower + projector over
### 2,000 training mixes, fits a linear per-frame degraded/clean probe at two
### taps (encoder output, post-projector), recovers intervals from the probe
### scores, and scores them with t-IoU against the audio-blind baselines.
###
### Probes TWO checkpoints in one job:
###   1. base Qwen/Qwen2-Audio-7B        (what pretraining provides)
###   2. models/sft_temporal_gc_timeaudio_h100 (what SFT shaped, 28615749)
###
### High probe t-IoU  -> information present, loss-level fixes can work.
### Probe near chance -> no text-side loss can fix it; pivot task/data.
###
### Mem check (per-core rusage rule): rusage[mem=24GB] x -n 4 = 96 GB total,
### within the gpuh100 ~720 GB node. OK.
###
### Submit with: bsub < jobs/analysis/frame_probe_temporal.sh
### ============================================================

#BSUB -q gpuh100
#BSUB -J frame-probe-temporal
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 4:00
#BSUB -o logs/frame_probe_temporal_%J.out
#BSUB -e logs/frame_probe_temporal_%J.err

set -euo pipefail

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_DIR}"
cd "$PROJECT_DIR"

mkdir -p "$EXPERIMENT_DIR/logs" "$EXPERIMENT_DIR/results/analysis"
module load cuda/11.8 || true
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTHONPATH=src
export HF_HOME="$EXPERIMENT_DIR/.cache/huggingface"

JSON_PATH="data/processed/temporal/train_nisqa_temporal_global_caption_aug_anchoroffset.json"
SFT_MODEL="$PROJECT_DIR/models/sft_temporal_gc_timeaudio_h100"

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Data     : $JSON_PATH"
echo "Models   : Qwen/Qwen2-Audio-7B + $SFT_MODEL"
echo "Started  : $(date)"
echo "=========================================="
nvidia-smi

if [ ! -f "$JSON_PATH" ]; then
  echo "ERROR: missing dataset $JSON_PATH"
  exit 1
fi
if [ ! -f "$SFT_MODEL/model.safetensors.index.json" ]; then
  echo "ERROR: no checkpoint index at $SFT_MODEL"
  exit 1
fi

echo "---- probe 1/2: base Qwen2-Audio-7B ----"
uv run python scripts/analysis/replication/probe_temporal_frames.py \
  --model-path "Qwen/Qwen2-Audio-7B" \
  --json-path "$JSON_PATH" \
  --data-root data \
  --max-samples 2000 \
  --val-fraction 0.2 \
  --batch-size 8 \
  --epochs 3 \
  --output-dir "$EXPERIMENT_DIR/results/analysis/frame_probe_base"

echo "---- probe 2/2: gc-timeaudio SFT checkpoint ----"
uv run python scripts/analysis/replication/probe_temporal_frames.py \
  --model-path "$SFT_MODEL" \
  --json-path "$JSON_PATH" \
  --data-root data \
  --max-samples 2000 \
  --val-fraction 0.2 \
  --batch-size 8 \
  --epochs 3 \
  --output-dir "$EXPERIMENT_DIR/results/analysis/frame_probe_gc_timeaudio"

echo "=========================================="
echo "Frame probe complete: $(date)"
echo "Results:"
echo "  $EXPERIMENT_DIR/results/analysis/frame_probe_base/probe_results.json"
echo "  $EXPERIMENT_DIR/results/analysis/frame_probe_gc_timeaudio/probe_results.json"
echo "=========================================="
