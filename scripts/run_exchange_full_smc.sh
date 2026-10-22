#!/bin/bash
set -euo pipefail
DECLARATION=${1:?declaration required}
HARNESS=${2:?frozen harness directory required}
OUTPUT=${3:?output directory required}
MODE=${4:-pair}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPYCACHEPREFIX=/tmp/sbsi-full-exchange-${SLURM_JOB_ID:?scheduled job required}
unset PYTHONPATH
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cd "$HARNESS"
sha256sum --check SHA256SUMS --quiet
test "$(realpath "$DECLARATION")" = "$HARNESS/declaration.json"
mkdir -p "$OUTPUT"
if [ "$MODE" = pair ]; then
    test "${SLURM_JOB_NODELIST}" = th-cl-nv02
    COST="$OUTPUT/pair_${SLURM_ARRAY_TASK_ID}_process_cost.json"
    test ! -e "$COST"
    exec /usr/bin/time -q -o "$COST" -f '{"seconds": %e, "exit_code": %x}' \
      "$PYTHON" "$HARNESS/benchmark_exchange_full_smc.py" run-pair --declaration "$DECLARATION" --replica "$SLURM_ARRAY_TASK_ID"
else
    test "$MODE" = report
    COST="$OUTPUT/report_process_cost.json"
    test ! -e "$COST"
    /usr/bin/time -q -o "$COST" -f '{"seconds": %e, "exit_code": %x}' \
      "$PYTHON" "$HARNESS/benchmark_exchange_full_smc.py" report --declaration "$DECLARATION" --output "$OUTPUT/report.json"
    "$PYTHON" "$HARNESS/benchmark_exchange_full_smc.py" finalize --output "$OUTPUT/report.json" --cost "$COST"
fi
