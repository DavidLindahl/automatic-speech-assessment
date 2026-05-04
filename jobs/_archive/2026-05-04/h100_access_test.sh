#!/bin/sh
#BSUB -q gpuh100
#BSUB -J h100-access-test
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 00:15
#BSUB -o logs/h100_access_test_%J.out
#BSUB -e logs/h100_access_test_%J.err

set -euo pipefail
module load cuda/11.8 || true

echo "=========================================="
echo "Job ID   : $LSB_JOBID"
echo "Host     : $(hostname)"
echo "Started  : $(date)"
echo "=========================================="

nvidia-smi
