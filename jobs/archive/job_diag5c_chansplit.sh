#!/bin/bash
#SBATCH --job-name=diag5c_chansplit
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_chansplit_%j.out
# CHANNEL SPLIT of d phi / d gamma: mean head vs density shape (hypothesis B).
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --gpus-per-node=a40-16gb:1 --mem=24G \
#       --cpus-per-task=8 jobs/job_diag5c_chansplit.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF
NGAL=${NGAL:-4000}; NNODE=${NNODE:-2000}; OFF=${OFF:-20000}; GGS=${GGS:-0.0 0.05}
DELTAS=${DELTAS:-0.01,0.005}
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
for GG in $GGS; do
  echo; echo "##################### g_true = $GG #####################"
  stdbuf -oL -eL python -u scripts/diag5c_chansplit.py \
    --n-gal "$NGAL" --n-node "$NNODE" --gal-offset "$OFF" --gamma "$GG" \
    --deltas "$DELTAS" ${EXTRA:-} 2>&1 | grep --line-buffered -vE "module command"
done
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
