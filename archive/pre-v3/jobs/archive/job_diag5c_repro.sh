#!/bin/bash
#SBATCH --job-name=diag5c_repro
#SBATCH --time=12:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_repro_%j.out
#
# Split Var(s) into SIGNAL / REPRODUCIBLE-BUT-SHEAR-BLIND / BANK NOISE at ONE setting.
# The 5C attenuation is I/Var(s)-1 with a sound numerator, so everything hinges on which
# part of the inflated denominator is removable.  Two disjoint half-banks are column slices
# of the same phi array, so Cov(s_A,s_B) isolates the reproducible part for free.  The K
# ladder checks the 1/K noise assumption; the channel lines say whether the blind part
# enters through the flow or through detection.  Supersedes the cross-run combination in
# the cont.168 discussion, which mixed two jobs at different settings.
#
# Submit:
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_repro.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF

NGAL=${NGAL:-20000}
NODES=${NODES:-2000,6000,20000}
GAMMA=${GAMMA:-0.05}
DELTA=${DELTA:-0.01}
CHUNK=${CHUNK:-24}
SEED=${SEED:-11}
BOOT=${BOOT:-400}

date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/diag5c_repro.py \
  --n-gal "$NGAL" --n-nodes "$NODES" --gamma "$GAMMA" --delta "$DELTA" \
  --chunk "$CHUNK" --seed "$SEED" --n-boot "$BOOT" ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
