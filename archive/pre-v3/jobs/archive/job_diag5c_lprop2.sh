#!/bin/bash
#SBATCH --job-name=d5c_lprop2
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/d5c_lprop2_%j.out
# Round 2 of the localised proposal, sized properly: 10x the galaxies (so the response test
# has a ~2%-of-truth error bar instead of ~6%) and a K ladder extended to 20000.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF
TMP=/home/z/Zekang.Zhang/.claude/jobs/be56a7ad/tmp
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/diag5c_localprop.py \
  --n-gal ${NGAL:-20000} --pool ${POOL:-100000} --n-nodes ${NODES:-500,2000,6000,20000} \
  --gammas ${GAMMAS:-0,0.05} --arms ${ARMS:-uniform,a0.2f0.05,a0.2f0.01} \
  --boot 400 --out "$TMP/localprop2_${SLURM_JOB_ID}.json" ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
