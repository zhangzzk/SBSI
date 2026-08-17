#!/bin/bash
#SBATCH --job-name=sc_bench
#SBATCH --time=01:00:00
#SBATCH --output=/home/z/Zekang.Zhang/logs/sc_bench_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/sc_bench_%j.err
#
# Is the score pass CPU-viable?  The `cip` QOS caps this account at 3 GPUs and 4 jobs, which
# is what is holding the row shards; the `cluster` QOS allows 3000 CPUs across 50 jobs.  So if
# a CPU node is within roughly 15x of an a40-16gb slice, `cluster` wins on throughput by sheer
# job count even though it loses badly per job.  This measures that ratio on IDENTICAL work --
# same rows, same legs, same node bank -- rather than guessing it.
#
# Deliberately small: the point is rows/second, not a science number, so nothing is saved and
# the Pi block is cut to the bone.  Submit the same script to both partitions.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
case "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" in
  *[0-9]Q|*[0-9]A|*[0-9]B) unset PYTORCH_CUDA_ALLOC_CONF ;;
esac
# BLAS threading is the whole game on CPU and pytorch will NOT pick it up from Slurm on its
# own; without this it runs single-threaded and the benchmark would slander the CPU path.
NT=${NTHREADS:-${SLURM_CPUS_PER_TASK:-8}}
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT
ROWS=${ROWS:-100000}
echo "host=$(hostname)  device=${DEV:-auto}  threads=$NT  rows=$ROWS"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
T0=$(date +%s)
python -u scripts/eval_score_select.py \
  --closure-g 0.05 --cut-abs-ehat 0.6 --max-rows "$ROWS" \
  --pi-rows 8192 --pi-samples 2 --pi-reps 1 \
  --ring none --shape-reps 1 --jk-blocks 20 \
  ${DEV:+--device "$DEV"} 2>&1 | grep -vE "module command|Pi rep "
T1=$(date +%s)
echo "### ELAPSED $((T1-T0)) s for $ROWS rows -> $(python -c "print(f'{$ROWS/max($T1-$T0,1):.1f}')") rows/s"
