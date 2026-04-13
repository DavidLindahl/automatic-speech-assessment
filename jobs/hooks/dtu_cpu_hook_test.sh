#!/bin/sh
### ============================================================
### DTU HPC - CPU callback relay test
### Purpose: validate end-of-job callback path through public relay
### ============================================================

#BSUB -q hpc
#BSUB -J thor-hook-test
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -M 1GB
#BSUB -W 00:10
#BSUB -o logs/thor_hook_test_%J.out
#BSUB -e logs/thor_hook_test_%J.err

set -eu

PROJECT_DIR="/work3/s234817/automatic-speech-assessment"
cd "$PROJECT_DIR"
mkdir -p logs

RELAY_URL="https://mac-servers.tailccbbb9.ts.net:8443/dtu-job"
RELAY_TOKEN="2dc7b1b0df08dd349d957c77e230350bfa4fc8f86aefcb4d"

notify_exit() {
  code="$1"
  status="success"
  if [ "$code" -ne 0 ]; then
    status="failed"
  fi

  payload=$(cat <<JSON
{"job_id":"${LSB_JOBID:-unknown}","job_name":"${LSB_JOBNAME:-thor-hook-test}","status":"${status}","queue":"hpc","host":"$(hostname)"}
JSON
)

  curl --max-time 10 --fail --silent --show-error \
    -X POST "$RELAY_URL" \
    -H "Authorization: Bearer $RELAY_TOKEN" \
    -H "Content-Type: application/json" \
    -d "$payload" || true
}

trap 'code=$?; notify_exit "$code"' EXIT

echo "CPU callback relay test start: $(date)"
echo "Job ID: ${LSB_JOBID:-unknown}"
echo "Host: $(hostname)"
sleep 10
echo "CPU callback relay test done: $(date)"
