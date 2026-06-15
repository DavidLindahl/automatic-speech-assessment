#!/bin/sh
### ============================================================
### DTU HPC LSF job script — build the PLAIN <|s|> timestamp-LAST temporal
### test sets (the basic-setup twins for evaluating sft_gc_timelast_plain_h100).
###
### WHY: the timestamp-last test sets were frozen only in the anchoroffset
### <a><f> format (test_{FOR,LIVE,P501}_temporal_global_caption_timelast_
### anchoroffset.json, commit eab8128), because the only timestamp-last model so
### far was ARM A (TimeAudio). The basic timestamp-last arm emits free-text
### <|seconds|> timestamps and must be scored on a <|s|> test set whose answer
### order matches what it was trained on. Those plain timelast test sets do not
### exist yet — this builds them.
###
### Exact mirror of the eab8128 anchoroffset-twin recipe, differing ONLY in
### --label-style:
###   --input-jsonl   the frozen timestamp-FIRST anchoroffset test set (source of
###                   the audio paths, intervals, MOS, filename_deg — all reused
###                   verbatim; only query+response are rewritten)
###   --caption-jsonl data/processed/eval/test_{X}.json (restores the full global
###                   caption verbatim, so the caption-first text is whole)
###   --label-style   global-caption-timelast   (plain <|s|>, caption first,
###                   temporal clause appended last)
###
### Output filenames drop the _anchoroffset suffix to match the plain
### timestamp-first test sets (test_{X}_temporal_global_caption.json):
###   test_{FOR,LIVE,P501}_temporal_global_caption_timelast.json
###
### CPU-only relabel (no GPU, no model). Submit with:
###   bsub < jobs/evaluate/build_timelast_plain_test_sets.sh
### ============================================================

#BSUB -q hpc
#BSUB -J build-timelast-plain-tests
#BSUB -n 2
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 01:00
#BSUB -o logs/build_timelast_plain_tests_%J.out
#BSUB -e logs/build_timelast_plain_tests_%J.err

set -eu

PROJECT_DIR="${PROJECT_DIR:-/work3/s234817/automatic-speech-assessment}"
cd "$PROJECT_DIR"

mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

# Same temporal query string used to build every other temporal train/test set,
# so train and eval prompts match.
QUERY="Please describe and evaluate the synthetic speech, and identify when the degradation occurs.<audio>"

echo "=========================================="
echo "Job ID   : ${LSB_JOBID:-local}"
echo "Host     : $(hostname)"
echo "Started  : $(date)"
echo "Style    : global-caption-timelast (plain <|s|>, timestamp last)"
echo "Query    : $QUERY"
echo "=========================================="

build_set() {
  set_name="$1"      # FOR / LIVE / P501
  # Source = frozen timestamp-first anchoroffset test set (audio/intervals/MOS).
  in_jsonl="data/processed/temporal/test_${set_name}_temporal_global_caption_anchoroffset.json"
  # Caption join = the global-MOS eval set, to restore the full caption verbatim.
  caption_jsonl="data/processed/eval/test_${set_name}.json"
  # Output = plain <|s|> timelast twin (no _anchoroffset suffix).
  out_jsonl="data/processed/temporal/test_${set_name}_temporal_global_caption_timelast.json"

  echo "------------------------------------------"
  echo "Building $set_name -> $out_jsonl"

  if [ ! -f "$in_jsonl" ]; then
    echo "ERROR: missing source test set $in_jsonl"
    exit 1
  fi
  if [ ! -f "$caption_jsonl" ]; then
    echo "ERROR: missing caption-join set $caption_jsonl"
    exit 1
  fi

  in_count="$(wc -l < "$in_jsonl" | tr -d ' ')"

  uv run python scripts/data/build_nisqa_temporal_json.py \
    --input-jsonl "$in_jsonl" \
    --caption-jsonl "$caption_jsonl" \
    --output-jsonl "$out_jsonl" \
    --label-style global-caption-timelast \
    --query "$QUERY"

  out_count="$(wc -l < "$out_jsonl" | tr -d ' ')"
  echo "Source records: $in_count  ->  wrote: $out_count"
  if [ "$in_count" != "$out_count" ]; then
    echo "ERROR: record count $out_count != source $in_count for $out_jsonl"
    exit 1
  fi
  echo "First record:"
  head -n 1 "$out_jsonl"
}

build_set "FOR"
build_set "LIVE"
build_set "P501"

echo "=========================================="
echo "Done: $(date)"
echo "Built 3 plain <|s|> timestamp-last test sets:"
echo "  data/processed/temporal/test_{FOR,LIVE,P501}_temporal_global_caption_timelast.json"
echo "Watch the per-set 'Caption-index misses' line in the output — expect 0,"
echo "matching the anchoroffset twins (eab8128)."
echo "=========================================="
