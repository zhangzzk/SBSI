#!/bin/bash
#SBATCH --job-name=diag5c_selfmix
#SBATCH --time=20:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_selfmix_%j.out
#
# SELF-CONSISTENCY test of the 5C Lagrangian estimator: draw the data from the estimator's
# OWN K-component mixture (uniform node index from the SAME bank, xhat ~ p(.|S_g z_k),
# detection Bernoulli(Pdet(S_g z_k))), so the mixture IS the data-generating density by
# construction and Bartlett's identity is mathematically forced.  The ordinary
# marginal-draw closure runs side by side at identical K / N_gal / seed as the control.
#
# Submit:
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_selfmix.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF

NGAL=${NGAL:-40000}
NODES=${NODES:-500,2000,6000}
GAMMAS=${GAMMAS:-0,0.05}
DELTA=${DELTA:-0.01}
SEED=${SEED:-11}
BOOT=${BOOT:-400}
DRAWS=${DRAWS:-mixture,marginal}
# PIN the marginal-draw galaxy sample across every K (and across jobs), so the control is
# the identical 40k rows at every rung and K is the only thing that moves.
OFFSET=${OFFSET:-20000}

date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/diag5c_selfmix.py \
  --n-gal "$NGAL" --n-nodes "$NODES" --gammas "$GAMMAS" --delta "$DELTA" \
  --gal-offset "$OFFSET" \
  --seed "$SEED" --boot "$BOOT" --draws "$DRAWS" ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
