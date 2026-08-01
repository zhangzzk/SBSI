#!/bin/bash
#SBATCH --job-name=diag5c_tailC
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_tailC_%j.out
#
# PROBE C: is the §5C score spread carried by nodes where the FLOW IS IN ITS TAIL?
# PART 1 bins every (galaxy, node) pair by its log-likelihood deficit phi0 - max_k phi0 and
# attributes BOTH Var_w(phi') (within-galaxy) and Var(s) (across-galaxy, the actual inflated
# denominator) to those bins exactly.  PART 2 sweeps the central-difference half-width to
# separate a real property of the flow from a finite-difference artefact.
# Setup mirrors diag5c_repro.py (job 15459872 / WORKLOG cont.169) exactly.
#
# Submit:
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_tailC.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF

NGAL=${NGAL:-20000}
NODES=${NODES:-2000,6000,20000}
DELTA=${DELTA:-0.01}
DELTAS=${DELTAS:-0.02,0.01,0.005,0.0025}
SK=${SK:-20000}
SNG=${SNG:-0}
CHUNK=${CHUNK:-24}
SEED=${SEED:-11}

date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/diag5c_tailC.py \
  --n-gal "$NGAL" --n-nodes "$NODES" --delta "$DELTA" --deltas "$DELTAS" \
  --stencil-k "$SK" --stencil-ngal "$SNG" --chunk "$CHUNK" --seed "$SEED" ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
