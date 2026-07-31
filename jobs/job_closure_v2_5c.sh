#!/bin/bash
#SBATCH --job-name=cl5c_v2
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/closure_v2_5c_%j.out
# INFERENCE.md 5C closure on the Gold-V2 joint forward model: draw the data FROM the model
# at a known shear and ask (5.8) to return it.  Model and data agree by construction, so
# this cannot be quadrature-limited the way the V1 agreement check was -- and it is the
# first test to exercise the detection channel and the population terms.
#
#   ROWS/NODES/GAMMA are the knobs; the delta sweep is the self-check.
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --gpus-per-node=a40-16gb:1 jobs/job_closure_v2_5c.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF
NGAL=${NGAL:-2000}; NODES=${NODES:-400 2000 10000 40000}; GAMMA=${GAMMA:-0.1}
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
for NN in $NODES; do
 for GG in $GAMMA; do
  echo; echo "########## n_node=$NN  gamma=$GG ##########"
  stdbuf -oL -eL python -u scripts/closure_v2_lagrangian.py \
    --n-gal "$NGAL" --n-node "$NN" --gamma "$GG" ${EXTRA:-} 2>&1 \
    | grep --line-buffered -vE "module command"
 done
done
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
