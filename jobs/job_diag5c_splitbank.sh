#!/bin/bash
#SBATCH --job-name=d5c_split
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_splitbank_%j.out
# Split-bank decomposition of Var(s) for the 5C Lagrangian score estimator.
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_splitbank.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF
NGAL=${NGAL:-2000}; OFF=${OFF:-16000}; KS=${KS:-1000,4000,16000}; GG=${GG:-0,0.05}
MODES=${MODES:-group block random interleave}
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
for MODE in $MODES; do
 echo; echo "########## split-mode = $MODE ##########"
 stdbuf -oL -eL python -u scripts/diag5c_splitbank.py \
   --n-gal "$NGAL" --gal-offset "$OFF" --n-nodes "$KS" --gammas "$GG" \
   --split-mode "$MODE" ${EXTRA:-} 2>&1 \
   | grep --line-buffered -vE "module command"
done
# deeper ladder: does the bank NOISE fall like 1/K, or keep growing?  Separate galaxy
# offset (must be >= max K), so it is its own internally-pinned ladder.
if [ "${DEEP:-1}" = "1" ]; then
 echo; echo "########## deep ladder (own galaxy sample, offset 40000) ##########"
 stdbuf -oL -eL python -u scripts/diag5c_splitbank.py \
   --n-gal "$NGAL" --gal-offset 40000 --n-nodes 2500,10000,40000 --gammas "$GG" \
   --split-mode group ${EXTRA:-} 2>&1 \
   | grep --line-buffered -vE "module command"
fi
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
