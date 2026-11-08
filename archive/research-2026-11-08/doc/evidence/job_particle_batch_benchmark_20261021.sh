#!/bin/bash
# Preparation only. Submit explicitly after checking the shared four-GPU cap.
# Example: sbatch --partition=inter --nodelist=th-cl-nv02 --output=/external/run/%j.out THIS_SCRIPT
#SBATCH --job-name=smc-batch-bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX=/tmp/smc-batch-bench-${SLURM_JOB_ID}
unset PYTHONPATH
SBSI_BENCH_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021
SBSI_BENCH_SOURCE=${SBSI_BENCH_SOURCE:-${SBSI_BENCH_ROOT}/source_v5b}
SBSI_BENCH_RUN=${SBSI_BENCH_RUN:-${SBSI_BENCH_ROOT}/dev_v5_0}
SBSI_BENCH_OUTPUT=${SBSI_BENCH_OUTPUT:?Set a fresh external JSON result path before submission}
SBSI_BENCH_SCRIPT=${SBSI_BENCH_SCRIPT:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/smc-bridge-audit-20261021/doc/evidence/benchmark_particle_batch_20261021.py}
export SBSI_CACHE_DIR=${SBSI_CACHE_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches}
export BLENDEMU_RUNS_DIR=${BLENDEMU_RUNS_DIR:-/project/ls-gruen/users/zekang.zhang/blendemu_runs}
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python "$SBSI_BENCH_SCRIPT" \
    --source "$SBSI_BENCH_SOURCE" --run "$SBSI_BENCH_RUN" --output "$SBSI_BENCH_OUTPUT" \
    --require-gpu-model V100 --warmup 1 --repetitions 5
