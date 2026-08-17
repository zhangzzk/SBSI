#!/bin/bash
#SBATCH --job-name=diag5c_kladder
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_kladder_%j.out
#
# UNCONFOUNDED k-ladder for the INFERENCE.md 5C Lagrangian score estimator.
#
# The earlier ladder moved the galaxy sample with K (gal_df started at row n_node), so the
# Var(s) growth 42.9 -> 93.0 could not be attributed to the bank.  Here --gal-offset is
# PINNED at 20000 (>= the largest K) for every rung, so all runs see the IDENTICAL 2000
# galaxies and only the node bank changes.  Note span = max(n_gal+n_node,
# gal_offset+n_gal) = 22000 for every rung, so load_rows also reads the identical raw
# block -- the galaxy frame is byte-identical across the ladder.
#
# Submit:
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_kladder.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF

NGAL=${NGAL:-2000}
OFFSET=${OFFSET:-20000}
NODES=${NODES:-500 1000 2000 5000 10000 20000}
GAMMA=${GAMMA:-0.0 0.05}
DELTAS=${DELTAS:-0.01}
SEED=${SEED:-11}

date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "ladder: n_gal=$NGAL gal_offset=$OFFSET seed=$SEED deltas=$DELTAS"
for NN in $NODES; do
 for GG in $GAMMA; do
  echo; echo "########## n_node=$NN  gamma=$GG  gal_offset=$OFFSET ##########"
  stdbuf -oL -eL python -u scripts/closure_v2_lagrangian.py \
    --n-gal "$NGAL" --n-node "$NN" --gal-offset "$OFFSET" \
    --gamma "$GG" --deltas "$DELTAS" --seed "$SEED" ${EXTRA:-} 2>&1 \
    | grep --line-buffered -vE "module command"
 done
done
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
