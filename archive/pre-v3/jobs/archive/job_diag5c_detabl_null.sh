#!/bin/bash
#SBATCH --job-name=d5c_detN
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_detabl_null_%j.out
# (3) the NULL with real statistics: N_gal = 20000 so N_detected ~ 10000.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF
date
for GG in 0.0 0.05; do
 echo; echo "########## NULL n_gal=20000 n_node=2000 gamma=$GG ##########"
 stdbuf -oL -eL python -u scripts/diag5c_detablation.py \
   --n-gal 20000 --n-node 2000 --gamma "$GG" --gal-offset 40000 --deltas 0.01 \
   2>&1 | grep -vE "module command"
done
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
