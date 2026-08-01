#!/bin/bash
#SBATCH --job-name=d5c_sigreach
#SBATCH --time=01:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/d5c_sigreach_%j.out
# How much shear signal reaches the V2 joint forward model at all (inputs -> context ->
# mean head -> phi).  Includes an independent reproduction of the certified <R_flow>.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/diag5c_signalreach.py \
  --n-gal ${NGAL:-20000} --n-node ${NODES:-2000} --n-gal-phi ${NGALPHI:-400} \
  --gammas ${GAMMAS:-0.01,0.05} ${EXTRA:-} 2>&1 | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
