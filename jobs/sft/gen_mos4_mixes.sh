#!/bin/bash
#BSUB -q milan
#BSUB -J gen-mos4-mixes
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 12:00
#BSUB -o /work3/s234817/automatic-speech-assessment/logs/gen_mos4_%J.out
#BSUB -e /work3/s234817/automatic-speech-assessment/logs/gen_mos4_%J.err
set -euo pipefail
cd /work3/s234817/automatic-speech-assessment
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
echo "Start: $(date) on $(hostname)"
# Flags matched to the MOS<=3 augmented set (generate_nisqa_temporal_augmented.sh)
# so the ONLY difference from the 13495 headline set is the MOS threshold.
python scripts/data/generate_nisqa_sim_lowmos_active.py \
  --mos-max-threshold 4.0 \
  --total-mix-files 0 \
  --seed 42 \
  --output-dir data/processed/nisqa_sim_mix_lowmos_active_mos4 \
  --allow-source-reuse \
  --max-source-passes 25 \
  --max-row-attempts 12 \
  --segment-min-seconds 0.6 \
  --segment-max-ratio 0.25 \
  --output-active-fraction-min 0.40 \
  --overwrite
echo "Done: $(date)"; ls data/processed/nisqa_sim_mix_lowmos_active_mos4/*.wav | wc -l
