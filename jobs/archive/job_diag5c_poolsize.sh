#!/bin/bash
#SBATCH --job-name=d5c_poolsz
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/d5c_poolsz_%j.out
# Exact enumeration of the whole population at M = 10k..300k, galaxies and data fixed:
# does the shear response grow with the size of the discrete population?
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF
TMP=/home/z/Zekang.Zhang/.claude/jobs/be56a7ad/tmp
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "########## SMOKE ##########"
stdbuf -oL -eL python -u scripts/diag5c_poolsize.py --n-gal 300 --n-sub 40 \
  --pools 4000,8000 --gal-offset 8000 --boot 100 2>&1 \
  | grep --line-buffered -vE "module command"
RC=$?; echo "### smoke exit $RC ###"
[ "$RC" -ne 0 ] && { date; echo "### ABORT ###"; exit "$RC"; }
echo; echo "########## PRODUCTION ##########"
stdbuf -oL -eL python -u scripts/diag5c_poolsize.py \
  --n-gal ${NGAL:-3000} --n-sub ${NSUB:-1200} --pools ${POOLS:-10000,30000,100000,300000} \
  --gal-offset ${GOFF:-300000} --out "$TMP/poolsize_${SLURM_JOB_ID}.json" ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
