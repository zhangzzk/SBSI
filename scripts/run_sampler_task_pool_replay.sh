#!/bin/bash
# No scheduler submission. Run inside one V100/eight-physical-core allocation.
set -euo pipefail
HARNESS=${1:?immutable prepared replay directory required}
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PYTHONPYCACHEPREFIX LD_PRELOAD
cd "$HARNESS"
sha256sum --check SHA256SUMS --quiet
mapfile -t PARAMETERS < <("$PYTHON" -c 'import json,sys;d=json.load(open(sys.argv[1]));print("\n".join(d[k] for k in ("manifest","manifest_sha256","pool_process_cost","audit_process_cost","report")))' "$HARNESS/replay.json")
test "${#PARAMETERS[@]}" = 5
bash "$HARNESS/run_sampler_task_pool.sh" "${PARAMETERS[0]}" "${PARAMETERS[1]}" "${PARAMETERS[2]}"
(set -o noclobber; : > "${PARAMETERS[3]}")
/usr/bin/time -q -o "${PARAMETERS[3]}" \
    -f '{"wall_seconds": %e, "user_seconds": %U, "system_seconds": %S, "max_rss_kib": %M, "exit_code": %x}' \
    "$PYTHON" "$HARNESS/benchmark_sampler_task_pool_replay.py" audit \
    --declaration "$HARNESS/replay.json" --output "${PARAMETERS[4]}"
"$PYTHON" "$HARNESS/benchmark_sampler_task_pool_replay.py" finalize \
    --output "${PARAMETERS[4]}" --cost "${PARAMETERS[3]}"
