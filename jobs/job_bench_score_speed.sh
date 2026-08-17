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
# `Pi` fast-path exactness check.  Pin the GRES to compare cards; the results are worth
# re-measuring whenever a new card appears, because precision safety is NOT portable across
# them (tf32 is fine on an A40 and biases an H200 by 7x the statistical error):
#
#   sbatch $(jobs/pick_gpu.sh) jobs/job_bench_score_speed.sh             # fastest card free
#   sbatch --partition=inter --gpus-per-node=h200nvl:1 jobs/job_bench_score_speed.sh
#   sbatch --partition=cip --gpus-per-node=a40-16gb:1 jobs/job_bench_score_speed.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
# vGPU slices (their names end in a profile letter, e.g. "NVIDIA A40-16Q") lack the CUDA VMM
# APIs the expandable allocator needs.  Detected from the card we actually got, rather than
# passed in as a flag, so that picking the card dynamically -- jobs/pick_gpu.sh -- cannot
# leave a stale setting behind or need the caller to remember one.
case "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" in
  *[0-9]Q|*[0-9]A|*[0-9]B) unset PYTORCH_CUDA_ALLOC_CONF ;;
esac

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
