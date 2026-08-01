#!/bin/bash
#SBATCH --job-name=diag5c_probeA3
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_probeA3_%j.out
#
# PROBE A confound check.  Every probe-A number came from ONE stencil width (delta = 0.01),
# and the |phi'| tail it found reaches 1e7 -- 2e5 nats of log-density between gamma = -+0.01
# for a single node, which is not credible as a smooth derivative.  A jump J inside the
# stencil gives phi' = J/(2 delta), so an artefact scales like 1/delta and a real derivative
# does not.  Same nodes, same galaxies, same xhat, three widths spanning a factor 8.
#
# Submit:
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_probeA3.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF

NGAL=${NGAL:-20000}
NODES=${NODES:-1000,2000,6000,20000}
DELTAS=${DELTAS:-0.02,0.01,0.0025}
CHUNK=${CHUNK:-24}
SEED=${SEED:-11}

date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/diag5c_probeA3_delta.py \
  --n-gal "$NGAL" --n-nodes "$NODES" --deltas "$DELTAS" --chunk "$CHUNK" \
  --seed "$SEED" ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
