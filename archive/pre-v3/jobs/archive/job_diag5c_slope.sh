#!/bin/bash
#SBATCH --job-name=d5c_slope
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/d5c_slope_%j.out
# Denominator-free response-slope scoreboard + the (5.9b) drift correction.
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_slope.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF
NGAL=${NGAL:-20000}; NODES=${NODES:-2000 10000}
GAMMAS=${GAMMAS:-0,0.01,0.02,0.05,0.10,0.20}
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
for NN in $NODES; do
  echo; echo "########## K=$NN  N_gal=$NGAL ##########"
  stdbuf -oL -eL python -u scripts/diag5c_slope.py \
    --n-gal "$NGAL" --n-node "$NN" --gammas "$GAMMAS" ${EXTRA:-} 2>&1 \
    | grep --line-buffered -vE "module command"
done
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
