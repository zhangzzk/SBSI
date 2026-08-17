#!/bin/bash
#SBATCH --job-name=d5c_lprop
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/d5c_lprop_%j.out
# Per-galaxy LOCALISED node proposal for the 5C Lagrangian estimator (hypothesis A).
# Runs a small SMOKE config first in the same allocation (GPU slots are contended), and
# only proceeds to the production ladder if it exits 0.
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_localprop.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF
TMP=/home/z/Zekang.Zhang/.claude/jobs/be56a7ad/tmp
NGAL=${NGAL:-2000}; POOL=${POOL:-100000}; NODES=${NODES:-500,2000,6000}
GAMMAS=${GAMMAS:-0,0.05}; ARMS=${ARMS:-uniform,a0.2f0.05,a0.2f0.01}
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "########## SMOKE ##########"
  stdbuf -oL -eL python -u scripts/diag5c_localprop.py \
    --n-gal 300 --pool 8000 --n-nodes 200,600 --gammas 0,0.05 \
    --arms uniform,a0.2f0.05 --boot 100 \
    --out "$TMP/localprop_smoke_${SLURM_JOB_ID}.json" 2>&1 \
    | grep --line-buffered -vE "module command"
  RC=$?
  echo "### smoke exit $RC ###"
  [ "$RC" -ne 0 ] && { date; echo "### ABORT: smoke failed ###"; exit "$RC"; }
fi

echo; echo "########## PRODUCTION  ngal=$NGAL pool=$POOL K=$NODES arms=$ARMS ##########"
stdbuf -oL -eL python -u scripts/diag5c_localprop.py \
  --n-gal "$NGAL" --pool "$POOL" --n-nodes "$NODES" --gammas "$GAMMAS" \
  --arms "$ARMS" --out "$TMP/localprop_${SLURM_JOB_ID}.json" ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
