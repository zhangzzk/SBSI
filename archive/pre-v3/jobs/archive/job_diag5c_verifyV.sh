#!/bin/bash
#SBATCH --job-name=diag5c_verifyV
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_verifyV_%j.out
#
# ADVERSARIAL VERIFICATION of probe C-flow-tail.
# PART A: is the deep-deficit bins' "0.00% of Var(s)" a real zero or a hidden cancellation?
#         (sd(s^b), max|s^b|, per-galaxy weight MAX, counterfactual Var with the tail deleted)
# PART B: the per-deficit-bin stencil table on ALL 9,918 galaxies, not the 3,000 subsample.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF

date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/diag5c_verifyV_tailcheck.py \
  --n-gal "${NGAL:-20000}" --n-nodes "${K:-20000}" --delta "${DELTA:-0.01}" \
  --chunk "${CHUNK:-24}" --seed "${SEED:-11}" --with-stencil 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
