#!/bin/bash
#SBATCH --job-name=diag5c_verifyR2
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_verifyR2_%j.out
#
# ADVERSARIAL CHECK of probe A.  Three untested steps behind its headline replacement claim
# ("Var_w/ESS is the wrong model; variance does not decompose over node blocks; ESS is not
# the effective sample size"):
#   1. Var_w/ESS is not the delta-method SNIS variance -- sum_k w^2 (phi'-s)^2 is.  Measure it.
#   2. batch means equal-weights the blocks; the pooled estimator weights them by W_m.
#      The predicted mismatch factor M*sum W_m^2 is computable exactly.
#   3. no error bar was quoted on any measured noise level; measure the bank-REALISATION
#      scatter directly on equal-size disjoint column blocks.
# Same layout as jobs 15459872 / 15461075 bank 0 (rows 0:20000, seed 11, delta 0.01).
#
# Submit:
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_verifyR2.sh
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

date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/diag5c_verifyR2_scene.py \
  --n-gal "$NGAL" --n-nodes "$NODES" --delta "$DELTA" \
  --chunk "$CHUNK" --seed "$SEED" ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
