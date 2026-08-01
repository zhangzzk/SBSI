#!/bin/bash
#SBATCH --job-name=d5c_detA
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_detabl_main_%j.out
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

echo "########## POPULATION TERMS, 200k scenes ##########"
stdbuf -oL -eL python -u scripts/diag5c_detablation.py \
  --n-pop 200000 --n-node 2000 --pop-only 2>&1 | grep -vE "module command"

for NN in 2000 10000; do
 for GG in 0.0 0.05; do
  echo; echo "########## ABLATION n_node=$NN gamma=$GG ##########"
  stdbuf -oL -eL python -u scripts/diag5c_detablation.py \
    --n-gal 2000 --n-node "$NN" --gamma "$GG" --gal-offset 40000 --deltas 0.01 \
    2>&1 | grep -vE "module command"
 done
done
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
