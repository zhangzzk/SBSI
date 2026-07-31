#!/bin/bash
#SBATCH --job-name=lag5c_sw
#SBATCH --time=03:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/lag5c_sweep_%j.out
# Does the Eulerian/Lagrangian disagreement shrink as the grid is refined?
#
# The identity is an integration by parts: exact for the integrals, but on a finite grid
# it is only as good as the quadrature.  If the flow is sharp in `e` the posterior sits on
# a few nodes and the DISCRETE identity fails -- which looks like a bug but is not one.
# ESS (printed per run) is the number that separates the two explanations.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF
MODEL=${MODEL:-models/measurement_flow_g0_ngmix_meas_szfl_noz_lam450_fixresp_s501.pt}
ROWS=${ROWS:-2000}
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
for GN in 41 61 81 121; do
  # keep deriv-chunk * G ~ 10k rows so the double-backward graph fits 16 GB
  case $GN in 41) DC=8;; 61) DC=4;; 81) DC=2;; 121) DC=1;; esac
  echo; echo "########## grid-n=$GN  deriv-chunk=$DC ##########"
  stdbuf -oL -eL python -u scripts/check_lagrangian_agreement.py \
    --measurement-model "$MODEL" --max-rows "$ROWS" --grid-n "$GN" \
    --slab 2000 --deriv-chunk "$DC" 2>&1 | grep --line-buffered -vE "module command"
done
date; echo "### DONE ###"
