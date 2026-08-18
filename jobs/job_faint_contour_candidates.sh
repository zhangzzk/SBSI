#!/bin/bash
#SBATCH --job-name=faint_scan
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/faint_scan_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/faint_scan_%j.err
#
# Which faint constgold galaxy draws the cleanest measurement contour?
#
#   sbatch jobs/job_faint_contour_candidates.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate py31
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI}
cd "$REPO" || exit 1
export BLENDEMU_ROOT=${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}
export PYTHONPATH="$REPO:$BLENDEMU_ROOT:$PYTHONPATH"

date
python -u jobs/diag_faint_contour_candidates.py ${EXTRA:-} 2>&1 | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
