#!/bin/bash
### ============================================================
### DTU HPC — Submit the paper-style warmup -> DPO -> eval pipeline
### Run from repo root on the HPC login node:
###   bash scripts/train/submit_dpo_paper_half_pipeline.sh
### ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$HOME/asa-paper-half}"

mkdir -p "$EXPERIMENT_DIR/logs"

submit_job() {
    script_path="$1"
    dependency="${2:-}"

    if [ -n "$dependency" ]; then
        output=$(bsub -w "$dependency" < "$script_path")
    else
        output=$(bsub < "$script_path")
    fi

    echo "$output" >&2
    echo "$output" | sed -n 's/Job <\([0-9][0-9]*\)>.*/\1/p'
}

cd "$EXPERIMENT_DIR"

WARMUP_JOB=$(submit_job "$PIPELINE_ROOT/jobs/global/sft/sft_warmup_paper_half_h100.sh")
WARMUP_EVAL_JOB=$(submit_job "$PIPELINE_ROOT/jobs/global/eval/evaluate_sft_warmup_paper_half_h100.sh" "done($WARMUP_JOB)")
GENERATE_JOB=$(submit_job "$PIPELINE_ROOT/jobs/global/data/generate_dpo_paper_half_h100.sh" "done($WARMUP_JOB)")
DPO_JOB=$(submit_job "$PIPELINE_ROOT/jobs/global/alld/dpo_paper_half_h100.sh" "done($GENERATE_JOB)")
GREEDY_EVAL_JOB=$(submit_job "$PIPELINE_ROOT/jobs/global/eval/evaluate_dpo_paper_half_h100_greedy.sh" "done($DPO_JOB)")
SAMPLED_EVAL_JOB=$(submit_job "$PIPELINE_ROOT/jobs/global/eval/evaluate_dpo_paper_half_h100_sampled.sh" "done($DPO_JOB)")

cat <<EOF
Submitted paper-style DPO pipeline:
  warmup:      $WARMUP_JOB
  warmup eval: $WARMUP_EVAL_JOB
  dpo data:    $GENERATE_JOB
  dpo train:   $DPO_JOB
  eval greedy: $GREEDY_EVAL_JOB
  eval sampled:$SAMPLED_EVAL_JOB
  outputs:     $EXPERIMENT_DIR
EOF
