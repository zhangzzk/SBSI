#!/bin/bash
# Run inside an already allocated one-V100/eight-physical-core srun step.
set -euo pipefail
MANIFEST=${1:?immutable manifest path required}
MANIFEST_SHA256=${2:?manifest SHA256 required}
OUTER_COST=${3:?fresh outer cost receipt outside the task output root required}
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
HARNESS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_sampler_task_pool.py
# Reserve only a fresh external cost file, before timing the complete Python CLI.
# This small launcher metadata check is outside the measured execution process.
"$PYTHON" - "$MANIFEST" "$MANIFEST_SHA256" "$OUTER_COST" <<'PYCODE'
import hashlib, json, os, sys
from pathlib import Path
manifest, expected, name = sys.argv[1:]
assert hashlib.sha256(Path(manifest).read_bytes()).hexdigest() == expected, 'changed manifest'
d = json.loads(Path(manifest).read_text())
cost = Path(name)
assert cost.is_absolute() and str(cost.resolve()) == name, 'canonical absolute cost path required'
assert not any(cost.is_relative_to(Path(root)) for root in
    [d['output_root'], *(v['path'] for v in d['sources'].values())]), 'cost must be outside outputs/source'
fd = os.open(cost, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.close(fd)
PYCODE
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PYTHONPYCACHEPREFIX LD_PRELOAD
/usr/bin/time -q -o "$OUTER_COST" \
    -f '{"wall_seconds": %e, "user_seconds": %U, "system_seconds": %S, "max_rss_kib": %M, "exit_code": %x}' \
    "$PYTHON" "$HARNESS" run --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA256"
