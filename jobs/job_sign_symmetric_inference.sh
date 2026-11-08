#!/bin/bash
#SBATCH --partition=inter
#SBATCH --gres=gpu:h200nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
H=@STUDY_ROOT@
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SBSI_CACHE_DIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export BLENDEMU_RUNS_DIR=/project/ls-gruen/users/zekang.zhang/blendemu_runs
unset PYTHONPATH
cd "$H/source_snapshot"
"$PY" "$H/verify.py"
"$PY" "$H/detect_resources.py" -o "$H/logs/resources_${SLURM_JOB_ID}.json"
case "$1" in
  preflight)
    "$PY" -u "$H/check.py" preflight
    ;;
  selection)
    test -f "$H/preflight_passed.json"
    side=original
    index=$SLURM_ARRAY_TASK_ID
    if (( index >= 4 )); then side=twin; index=$((index-4)); fi
    "$PY" -u -c 'import json,sys;from pathlib import Path;sys.path[:0]=[".","scripts"];from run_galsbi_bright import selection;root=Path(sys.argv[1]);selection(root,json.loads((root/"plan.json").read_text()),int(sys.argv[2]),reference=Path(sys.argv[3]))' \
      "$H/normalization/$side/it0" "$index" "$H/reference"
    ;;
  estimate-c1)
    "$PY" -I -u "$H/estimate.py" c1
    read -r g1 g2 < <("$PY" -I -c "import json;print(*(round(x,5) for x in json.load(open('$H/c1/estimate.json'))['g']))")
    c2=$(sbatch --parsable --gres=gpu:h200nvl:1 --job-name=signsym-c2 --output="$H/logs/c2_%j.out" \
      "$H/worker.sh" c2 "$g1" "$g2" 2028600001 2033600001)
    finish=$(sbatch --parsable --dependency="afterok:$c2" --kill-on-invalid-dep=yes --partition=cluster --gres=none \
      --cpus-per-task=4 --mem=64G --time=02:00:00 --job-name=signsym-final --output="$H/logs/final_%j.out" \
      "$H/pipeline.sh" finish)
    "$PY" - "$H/jobs.json" "$c2" "$finish" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); jobs=json.loads(p.read_text())
jobs.update(c2=sys.argv[2].split(';')[0], final=sys.argv[3].split(';')[0]); p.write_text(json.dumps(jobs,indent=2))
PY
    sbatch --parsable --dependency="after:$c2+2" --partition=cluster --gres=none --cpus-per-task=1 --mem=2G \
      --time=00:10:00 --job-name=signsym-c2-check --output="$H/logs/check_%j.out" "$H/pipeline.sh" check
    ;;
  finish)
    "$PY" -I -u "$H/estimate.py" c2
    "$PY" "$H/check.py"
    ;;
  check)
    "$PY" "$H/check.py"
    ;;
  *) echo "unknown stage: $1" >&2; exit 2 ;;
esac
