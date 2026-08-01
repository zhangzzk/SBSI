#!/bin/bash
#SBATCH --job-name=diag5c_probeA
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_probeA_%j.out
#
# PROBE A: does the weighted spread of the integrand grow with K fast enough to cancel the
# averaging gain?  Var_MC(s) ~= Var_w(phi')/ESS, so the bank-noise floor has two moving
# parts.  If Var_w(phi') ~ K^p with p ~ q (ESS ~ K^q) the noise is flat in K, which would
# explain cont.169's (a) half-banks agreeing at only 6-8% AND (b) Var(s) not falling when
# the bank doubles.  Nested bank ladder (one phi array, column slices), gamma = 0 only.
# Bank 0 reproduces job 15459872's layout exactly, so its Var(s)/corr/sigma^2_half are a
# built-in validity check; bank 1 is a disjoint pool for bank-to-bank scatter.
#
# Submit:
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_probeA.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF

NGAL=${NGAL:-20000}
NODES=${NODES:-1000,2000,6000,20000}
DELTA=${DELTA:-0.01}
CHUNK=${CHUNK:-24}
SEED=${SEED:-11}
BOOT=${BOOT:-400}
BANKS=${BANKS:-2}
BLOCK=${BLOCK:-10}
OUT=${OUT:-$REPO/data/diag5c_probeA_spread.npz}

date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/diag5c_probeA_spread.py \
  --n-gal "$NGAL" --n-nodes "$NODES" --delta "$DELTA" --chunk "$CHUNK" \
  --seed "$SEED" --n-boot "$BOOT" --n-banks "$BANKS" --n-block "$BLOCK" \
  --out "$OUT" ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
