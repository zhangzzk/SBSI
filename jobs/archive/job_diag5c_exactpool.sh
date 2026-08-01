#!/bin/bash
#SBATCH --job-name=d5c_expool
#SBATCH --time=10:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/d5c_expool_%j.out
# The K -> M limit: phi over the ENTIRE discrete population, plus every sampled arm read
# off the same exact block.  Smoke config first, in the same allocation.
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_exactpool.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF
TMP=/home/z/Zekang.Zhang/.claude/jobs/be56a7ad/tmp
NGAL=${NGAL:-2000}; POOL=${POOL:-100000}; NSUB=${NSUB:-200}
NODES=${NODES:-500,2000,6000}; GAMMAS=${GAMMAS:-0,0.05}
ARMS=${ARMS:-uniform,a0.2f0.05,a0.2f0.01}
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "########## SMOKE ##########"
  stdbuf -oL -eL python -u scripts/diag5c_exactpool.py \
    --n-gal 300 --pool 8000 --n-sub 40 --n-nodes 200,600 --gammas 0,0.05 \
    --arms uniform,a0.2f0.05 --boot 100 2>&1 | grep --line-buffered -vE "module command"
  RC=$?; echo "### smoke exit $RC ###"
  [ "$RC" -ne 0 ] && { date; echo "### ABORT: smoke failed ###"; exit "$RC"; }
fi

echo; echo "########## PRODUCTION ngal=$NGAL pool=$POOL nsub=$NSUB ##########"
stdbuf -oL -eL python -u scripts/diag5c_exactpool.py \
  --n-gal "$NGAL" --pool "$POOL" --n-sub "$NSUB" --n-nodes "$NODES" \
  --gammas "$GAMMAS" --arms "$ARMS" \
  --out "$TMP/exactpool_${SLURM_JOB_ID}.json" ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
