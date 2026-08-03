#!/bin/bash
#SBATCH --job-name=bench_speed
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --gpus-per-node=1
#SBATCH --output=/home/z/Zekang.Zhang/logs/bench_speed_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bench_speed_%j.err
#
# Cost profile for `eval_score_select.py`, run at a size that finishes in minutes, plus the
# `Pi` fast-path exactness check.  Override the GRES to separate "our arithmetic is
# inefficient" from "our GPU is a third of a card":
#
#   sbatch jobs/job_bench_score_speed.sh                                 # inter, any GPU
#   sbatch --gres=gpu:h200nvl:1 jobs/job_bench_score_speed.sh            # inter, H200
#   sbatch -p cip --gres=gpu:a40-16gb:1 --export=ALL,NO_EXPANDABLE_SEGMENTS=1 \
#          jobs/job_bench_score_speed.sh                                 # the old vGPU slice
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
# The A40-16Q vGPU slices lack the CUDA VMM APIs the expandable allocator needs.
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF

date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The exactness check first: if the Pi rewrite is wrong, its timings are irrelevant.
python -u scripts/check_pi_fastpath.py --precision "${PRECISION:-fp32}"
CHECK=$?

python -u scripts/bench_score_speed.py \
  --rows "${ROWS:-40000}" --pi-rows "${PIROWS:-8192}" --pi-nodes "${PINODES:-8}" \
  --modes "${MODES:-fp32,tf32,bf16,fp16}" ${EXTRA:-}
STATUS=$?
date; echo "### DONE (bench $STATUS, pi-check $CHECK) ###"; exit $((STATUS + CHECK))
