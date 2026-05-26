#!/usr/bin/env bash
# Lint LSF memory budgets across every jobs/**/*.sh script.
#
# Hard rule: LSF interprets `#BSUB -R "rusage[mem=X]"` together with
# `#BSUB -n N` as X * N total memory per job. Getting this wrong has
# triggered angry DTU IT emails (2026-04-30, 2026-05-04) and killed
# the entire pending queue. This linter is the firebreak.
#
# Caps per queue (verify against `nodestat -F <queue>` periodically):
#   gpuh100:  720 GB per node
#   gpua40:   384 GB per node  (conservative; nodes vary)
#   gpul40s:  384 GB per node  (conservative)
#   gpua10:   192 GB per node  (conservative)
#   hpc:      192 GB per node  (CPU-only, conservative)
#
# Exits non-zero on any violation. Run before submitting:
#   bash jobs/_lib/lint-budget.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Map queue -> total memory cap in GB.
queue_cap() {
    case "$1" in
        gpuh100) echo 720 ;;
        gpua100) echo 384 ;;
        gpua40)  echo 384 ;;
        gpul40s) echo 384 ;;
        gpua10)  echo 192 ;;
        hpc)     echo 192 ;;
        *)       echo "" ;;
    esac
}

violations=0
scripts_checked=0

# Walk every .sh under jobs/, including subdirectories.
# Exclude _archive/ and _lib/ themselves.
while IFS= read -r script; do
    case "$script" in
        jobs/_archive/*) continue ;;
        jobs/_lib/*)     continue ;;
    esac

    scripts_checked=$((scripts_checked + 1))

    queue=$(awk '/^#BSUB -q / {print $3; exit}' "$script")
    cores=$(awk '/^#BSUB -n / {print $3; exit}' "$script")
    # rusage[mem=NGB] -> N
    mem=$(awk -F'rusage\\[mem=' '/^#BSUB -R "rusage\[mem=/ {split($2, a, "GB"); print a[1]; exit}' "$script")
    # -M NGB -> N
    cap=$(awk '/^#BSUB -M / {gsub(/GB$/, "", $3); print $3; exit}' "$script")

    [ -z "$queue" ] && continue   # not a BSUB script
    [ -z "$cores" ] && cores=1
    [ -z "$mem" ] && continue     # no memory request, skip

    queue_max=$(queue_cap "$queue")
    if [ -z "$queue_max" ]; then
        echo "WARN: $script: unknown queue '$queue' — skipping cap check"
        continue
    fi

    total=$((cores * mem))

    if [ "$total" -gt "$queue_max" ]; then
        echo "FAIL: $script"
        echo "      queue=$queue, -n $cores, rusage[mem=${mem}GB] -> total ${total}GB > cap ${queue_max}GB"
        violations=$((violations + 1))
        continue
    fi

    # Also lint: rusage[mem] != -M (they should be the same per-core number).
    if [ -n "$cap" ] && [ "$cap" != "$mem" ]; then
        echo "FAIL: $script"
        echo "      -M ${cap}GB != rusage[mem=${mem}GB] — should match (both are per-core)"
        violations=$((violations + 1))
    fi
done < <(find jobs -name '*.sh' -type f -print)

echo ""
echo "lint-budget: checked $scripts_checked scripts; $violations violations"

[ "$violations" -eq 0 ]
