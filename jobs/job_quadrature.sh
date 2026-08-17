#!/bin/bash
#SBATCH --job-name=quadrature
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/quadrature_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/quadrature_%j.err
#
# Flow-free, data-free convergence test of the §5B population machinery.  CPU only -- no GPU
# is needed because there is no flow in it, which is the whole point.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
date
python -u scripts/check_quadrature.py ${EXTRA:-} 2>&1 | grep -v "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
