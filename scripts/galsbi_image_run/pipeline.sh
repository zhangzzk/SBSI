#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
# rule.md stages: reference | preflight (GPU) | selection (GPU array) | estimate-c1 | finish
set -euo pipefail
H=@RUN_DIR@  # set once per run (launch.sh header)
[ -f $H/rule.json ] || { echo "run directory not set: $H"; exit 1; }
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
rj() {  # rj <key> [<key> ...]: a value from rule.json (lists printed space-separated)
    "$PY" -I -c 'import json,sys
v = json.load(open(sys.argv[1]))
for k in sys.argv[2:]: v = v[k]
print(*v) if isinstance(v, list) else print(v)' "$H/rule.json" "$@"
}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
unset PYTHONPATH
cd "$H/source_snapshot"
sha256sum --check --quiet SHA256SUMS
run() {  # run a repository script from the frozen snapshot
    local script=$1; shift
    "$PY" -u -c 'import pathlib,runpy,sys;sys.path[:0]=[".","scripts"];import sbsi;assert pathlib.Path(sbsi.__file__).resolve().parent.parent==pathlib.Path.cwd();runpy.run_path(sys.argv.pop(1),run_name="__main__")' "$script" "$@"
}
case "$1" in
  reference)
    cases=($(rj cases))
    [ ${#cases[@]} -eq $((${cases[-1]}-${cases[0]}+1)) ] || { echo "rule.json cases must be a contiguous range"; exit 1; }
    run scripts/prepare_galsbi_image_reference.py --output "$H/reference" --cases ${cases[0]} $((${cases[-1]}+1)) \
      --fraction $(rj identity_fraction) --seed $(rj identity_seed)
    "$PY" -I -u "$H/study.py" units c1
    ;;
  preflight)
    P=$H/preflight
    mkdir -p "$P"
    bright=$(sed -n '1,2p' "$H/c1/plus_bright_ids.txt" | tr '\n' ' ')
    faint=$(sed -n '1,2p' "$H/c1/minus_faint_ids.txt" | tr '\n' ' ')
    short=(--steps 256 --warmup 128 --profile-stride 16 --device cuda)
    run scripts/run_galsbi_inference.py --output "$P/bright_g0" --reference "$H/reference/plus" --shear 0 0 \
      --seed 1 --initial-bank-seed 1 --observation-indices $bright "${short[@]}" \
      --betas 1 0.7883 0.6167 0.4449 0.2849 0.1473 0.05014 0.005 --narrow-step-probability .5
    run scripts/run_galsbi_inference.py --output "$P/faint_g" --reference "$H/reference/minus" --shear -0.02 0.001 \
      --seed 2 --initial-bank-seed 2 --observation-indices $faint "${short[@]}"
    mkdir -p "$P/selection"
    "$PY" -I -c "import json;p=json.load(open('$H/normalization/plan.json'));p.update(selection_chunks=1,selection_chunk_size=1024,selection_seed_base=1);json.dump(p,open('$P/selection/plan.json','w'),indent=1)"
    run scripts/run_galsbi_bright.py --root "$P/selection" --stage selection --replica 0 --reference "$H/reference/plus"
    "$PY" -I - "$P" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
P = Path(sys.argv[1])
out = {}
for name in ('bright_g0', 'faint_g'):
    r = json.loads((P/name/'result.json').read_text())
    with np.load(P/name/'local_profiles.npz') as z:
        finite = all(np.isfinite(z[k]).all() for k in z.files)
    assert r['status'] == 'completed_diagnostic' and finite, name
    out[name] = dict(elapsed_seconds=r['elapsed_seconds'], profiles_finite=finite, timing=r['timing_seconds'])
s = json.loads((P/'selection/selection0/result.json').read_text())
with np.load(P/'selection/selection0/chunks.npz') as z:
    v = z['values']
assert s['model'] == 'like-1.0-galsbi' and np.all(v > 0) and np.isfinite(v).all()
out['selection'] = dict(values=v[0, :, 0, 1].tolist(), seconds=s['process_seconds'])
(P.parent/'preflight_passed.json').write_text(json.dumps(out, indent=1)+'\n')
print('PREFLIGHT_PASSED', json.dumps(out))
PY
    ;;
  selection)
    run scripts/run_galsbi_bright.py --root "$H/normalization" --stage selection --replica "$SLURM_ARRAY_TASK_ID" \
      --reference "$H/reference/plus"
    ;;
  estimate-c1)
    "$PY" -I -u "$H/study.py" estimate c1 plus
    "$PY" -I -u "$H/study.py" estimate c1 minus
    "$PY" -I -u "$H/study.py" units c2
    ids=()
    prefix=$(rj job_prefix)
    for ((w=0; w<$(rj resources c2_workers); w++)); do
      ids+=($(sbatch --parsable -c $(rj resources cpus_per_worker) --mem $(rj resources mem_per_worker) \
              -J $prefix-c2 -o "$H/logs/c2_%j.out" "$H/worker.sh" c2 $(rj seeds c2)))
    done
    deps=$(IFS=:; echo "${ids[*]}")
    finish=$(sbatch --parsable --dependency="afterany:$deps" -J $prefix-finish -o "$H/logs/finish_%j.out" "$H/pipeline.sh" finish)
    echo "c2 workers ${ids[*]} finish $finish" | tee -a "$H/jobs.txt"
    ;;
  finish)
    "$PY" -I -u "$H/study.py" estimate c2 plus
    "$PY" -I -u "$H/study.py" estimate c2 minus
    "$PY" -I -u "$H/study.py" pair
    ;;
  *) echo "unknown stage: $1" >&2; exit 2 ;;
esac
