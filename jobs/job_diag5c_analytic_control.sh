#!/bin/bash
#SBATCH --job-name=d5c_anactl
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_analytic_control_%j.out
# ANALYTIC CONTROL for the 5C Lagrangian score estimator: same catalogue scenes, same shear
# map, same bank construction, same assembly helpers, same finite differences -- Gaussian
# emission instead of the trained flow.  CPU only; no torch compute is done.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
date
stdbuf -oL -eL python -u scripts/diag5c_analytic_control.py ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
