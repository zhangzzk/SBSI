#!/bin/bash
# Submit with one V100; the caller supplies the resource dependency explicitly.
set -euo pipefail
SOURCE=${1:?frozen source directory required}
RESULT=${2:?fresh output JSON required}
RUN=${3:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/sampler_search_20261021/dev_v5_0}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPYCACHEPREFIX=/tmp/sbsi-exchange-categorical-${SLURM_JOB_ID:?scheduled job required}
unset PYTHONPATH
cd "$SOURCE"
sha256sum --check SHA256SUMS --quiet
exec /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u \
    scripts/benchmark_exchange_trained.py --source "$SOURCE" \
    --manifest "$SOURCE/SHA256SUMS" --run "$RUN" --output "$RESULT" \
    --require-gpu-model V100
