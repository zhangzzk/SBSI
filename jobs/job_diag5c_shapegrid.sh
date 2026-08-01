#!/bin/bash
#SBATCH --job-name=diag5c_shapegrid
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_shapegrid_%j.out
#
# Does spending the node budget on SHAPE RESOLUTION fix the 5C information equality?
# Under primary_only shear, S_g moves exactly two of the ~18 scene coordinates (the
# primary's intrinsic e1,e2), so d/dg log p_hat is a directional derivative along the shape
# axes alone.  Budget ladder at FIXED K: arm A = K catalogue scenes as they come; arm B =
# M base scenes x (nr x na) STRATIFIED shape nodes, K = M*nr*na.  Trend in the equality
# ratio decides whether shear-direction coverage is the lever.  Bigger / self- / localised
# banks have already failed (cont.161, cont.167).
#
# Submit:
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_shapegrid.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF

NGAL=${NGAL:-20000}
K=${K:-20000}
# ladder from most shape resolution to least, all at K = 20000
SPLITS=${SPLITS:-100x20x10,200x10x10,1000x5x4,2000x5x2}
GAMMA=${GAMMA:-0.05}
DELTAS=${DELTAS:-0.01}
CHUNK=${CHUNK:-24}
SEED=${SEED:-11}
BOOT=${BOOT:-400}

date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/diag5c_shapegrid.py \
  --n-gal "$NGAL" --k "$K" --splits "$SPLITS" --gamma "$GAMMA" \
  --deltas "$DELTAS" --chunk "$CHUNK" --seed "$SEED" --n-boot "$BOOT" ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
