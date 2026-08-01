#!/bin/bash
#SBATCH --job-name=d5c_phian
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_phianatomy_%j.out
# phi-anatomy for the 5C Lagrangian score: where the Var_w(phi') mass sits as a function of
# CUMULATIVE POSTERIOR WEIGHT, whether phi'' is round-off dominated, and a float64 CPU
# precision control.  Run 1 reproduces the reference K=2000 galaxy sample (gal_offset=2000);
# runs 2-4 are a K ladder with the galaxies PINNED at offset 10000 so only the bank moves.
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_phianatomy.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

run () {   # $1 = n_node   $2 = gal_offset   $3 = chunk
  echo; echo "##################### n_node=$1  gal_offset=$2 #####################"
  stdbuf -oL -eL python -u scripts/diag5c_phianatomy.py \
    --n-gal ${NGAL:-2000} --n-node "$1" --gal-offset "$2" --chunk "$3" \
    --gamma ${GAMMA:-0.0} --n-f64 ${NF64:-50} ${EXTRA:-} 2>&1 \
    | grep --line-buffered -vE "module command"
}

run 2000 2000  250      # the reference configuration (matches job 15412224 K=2000 g=0)
run  500 10000 500      # K ladder, galaxies pinned
run 2000 10000 250
run 6000 10000 80

STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
