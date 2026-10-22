#!/bin/bash
# Run in one GPU/two physical CPU cores via srun --cpu-bind=cores.
set -euo pipefail
HARNESS=${1:?frozen harness directory required}
DECLARATION="$HARNESS/declaration.json"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
export PYTHONPYCACHEPREFIX=/tmp/sbsi-prior-reuse-${SLURM_JOB_ID:?scheduled job required}
unset PYTHONPATH
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cd "$HARNESS"
sha256sum --check SHA256SUMS --quiet
test "${SLURM_JOB_NODELIST}" = th-cl-nv02
test "${SLURM_CPUS_PER_TASK}" = 2
OUTPUT=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["output"])' "$DECLARATION")
test ! -e "$OUTPUT"
mkdir "$OUTPUT"
scontrol show job "$SLURM_JOB_ID" > "$OUTPUT/allocation.txt"
for pair in 0 1 2 3; do
    /usr/bin/time -q -o "$OUTPUT/pair_${pair}_process_cost.json" \
        -f '{"wall_seconds": %e, "user_seconds": %U, "system_seconds": %S, "max_rss_kib": %M, "exit_code": %x}' \
        "$PYTHON" "$HARNESS/benchmark_prior_reuse_smc.py" run-pair \
        --declaration "$DECLARATION" --replica "$pair"
done
/usr/bin/time -q -o "$OUTPUT/report_process_cost.json" \
    -f '{"wall_seconds": %e, "user_seconds": %U, "system_seconds": %S, "max_rss_kib": %M, "exit_code": %x}' \
    "$PYTHON" "$HARNESS/benchmark_prior_reuse_smc.py" report \
    --declaration "$DECLARATION" --output "$OUTPUT/report.json"
"$PYTHON" "$HARNESS/benchmark_prior_reuse_smc.py" finalize \
    --output "$OUTPUT/report.json" --cost "$OUTPUT/report_process_cost.json"
