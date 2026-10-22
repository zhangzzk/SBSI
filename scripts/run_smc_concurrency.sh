#!/bin/bash
# Run via srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores inside a
# one-GPU allocation requested with --hint=nomultithread. No MPS changes.
set -euo pipefail
HARNESS=${1:?frozen harness directory required}
DECLARATION="$HARNESS/declaration.json"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPYCACHEPREFIX=/tmp/sbsi-concurrency-${SLURM_JOB_ID:?scheduled job required}
unset PYTHONPATH
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cd "$HARNESS"
sha256sum --check SHA256SUMS --quiet
test "${SLURM_JOB_NODELIST}" = th-cl-nv02
test "${SLURM_CPUS_PER_TASK}" = 8
OUTPUT=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$DECLARATION")
test ! -e "$OUTPUT"
mkdir "$OUTPUT"
"$PYTHON" "$HARNESS/detect_resources.py" --output "$OUTPUT/resource_inventory.json"
scontrol show job "$SLURM_JOB_ID" > "$OUTPUT/allocation.txt"
for setting in 0 1 2 3 4 5; do
    /usr/bin/time -q -o "$OUTPUT/setting_${setting}_process_cost.json" \
        -f '{"wall_seconds": %e, "user_seconds": %U, "system_seconds": %S, "max_rss_kib": %M, "exit_code": %x}' \
        "$PYTHON" "$HARNESS/benchmark_smc_concurrency.py" run-setting \
        --declaration "$DECLARATION" --setting-index "$setting"
done
/usr/bin/time -q -o "$OUTPUT/report_process_cost.json" \
    -f '{"wall_seconds": %e, "user_seconds": %U, "system_seconds": %S, "max_rss_kib": %M, "exit_code": %x}' \
    "$PYTHON" "$HARNESS/benchmark_smc_concurrency.py" report \
    --declaration "$DECLARATION" --output "$OUTPUT/report.json"
