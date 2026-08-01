#!/bin/bash
#SBATCH --job-name=diag5c_probeA2
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_probeA2_%j.out
#
# PROBE A follow-up: job 15461075 refuted the p-q ~ 0 arithmetic (p-q = -0.51 +- 0.02) and
# showed the Var_w/ESS formula under-predicts the measured noise 6-10x, while the |phi'|
# BULK quantiles are stationary in K and only the MAX grows.  This measures the tail index
# (Hill), asks whether the extreme node is shared across galaxies, and runs the decisive
# causal test: winsorise |phi'| at a per-galaxy threshold FIXED at the smallest K and see
# whether the half-bank noise then falls like 1/K.  Clipping is a diagnostic only -- the
# clipped statistic is not the score.
#
# Submit:
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_probeA2.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF

NGAL=${NGAL:-20000}
NODES=${NODES:-1000,2000,6000,20000}
DELTA=${DELTA:-0.01}
CHUNK=${CHUNK:-24}
SEED=${SEED:-11}
BOOT=${BOOT:-400}
CLIPS=${CLIPS:-99.9,99,95}

date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/diag5c_probeA2_tail.py \
  --n-gal "$NGAL" --n-nodes "$NODES" --delta "$DELTA" --chunk "$CHUNK" \
  --seed "$SEED" --clips "$CLIPS" ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
