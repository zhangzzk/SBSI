#!/bin/bash
#SBATCH --job-name=lag5c
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/lag5c_%j.out
# INFERENCE.md 5C.5 cross-check (i): the Eulerian (2.2) and Lagrangian (5.5) scores
# must agree OBJECT BY OBJECT, since both equal d_gamma log p(xhat | gamma).  This is
# an identity, so it needs no statistics -- 100k rows is already decisive; more rows
# only sharpens the tail of the per-object scatter.
#
# A pass validates the reparametrization on the real flow.  It does NOT validate the
# full (5.8) estimator, which additionally needs P_det and the population terms that
# score_inference.py does not have -- that is stage 4.
#
# Submit e.g.:  ROWS=100000 sbatch jobs/job_lagrangian_check.sh
# `inter` also holds rtx2080ti (11G) and p5000 (16G) nodes, which cannot hold a slab.
# Pin the GPU type if the scheduler lands you on one:
#   sbatch --gpus-per-node=a40:1 jobs/job_lagrangian_check.sh
#
# On the `cip` vGPU slices set NO_EXPANDABLE_SEGMENTS=1 and resize -- those nodes have
# only 12 CPUs and 26-41 GB RAM, so the 128G above can never be satisfied there:
#   ROWS=20000 SLAB=5000 NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip \
#     --gres=gpu:a40-16gb:1 --mem=24G --cpus-per-task=8 jobs/job_lagrangian_check.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"

# expandable_segments needs the CUDA virtual-memory APIs, which the vGPU profiles
# (device reports as "NVIDIA A40-16Q") do not support -- any .to("cuda") then dies with
# `CUDA driver error: operation not supported`.
if [ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
  echo "NO_EXPANDABLE_SEGMENTS=1 -> PYTORCH_CUDA_ALLOC_CONF unset (cip vGPU)"
fi

ROWS=${ROWS:-100000}
GRID=${GRID:-61}
MODEL=${MODEL:-models/measurement_flow_g0_ngmix_meas_szfl_noz_lam450_fixresp_s501.pt}
SLAB=${SLAB:-20000}
CHUNK=${CHUNK:-1024}
EXTRA=${EXTRA:-}

echo "### ROWS=$ROWS GRID=$GRID MODEL=$(basename $MODEL) ###"; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/check_lagrangian_agreement.py \
  --measurement-model "$MODEL" --max-rows "$ROWS" --grid-n "$GRID" \
  --slab "$SLAB" --chunk "$CHUNK" \
  $EXTRA 2>&1 | grep --line-buffered -vE "module command"
STATUS=$?
date
echo "### DONE (exit $STATUS) ###"
exit $STATUS
