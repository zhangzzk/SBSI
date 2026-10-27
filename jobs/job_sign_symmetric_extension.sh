#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
H=@STUDY_ROOT@
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
cd "$H/source_snapshot"
"$PY" "$H/detect_resources.py" -o "$H/logs/resources_${SLURM_JOB_ID}.json"
case "$1" in
  prepare|preflight|check) "$PY" -u "$H/study.py" "$1" ;;
  estimate-c1)
    "$PY" -I -u "$H/study.py" c1
    read -r g1 g2 < <("$PY" -I -c "import json;print(*(round(x,5) for x in json.load(open('$H/c1/estimate.json'))['g']))")
    c2ids=()
    for index in 0 1 2 3; do
      id=$(sbatch --parsable --job-name=signsym500k-c2 --output="$H/logs/c2_%j.out" \
        "$H/worker.sh" c2 "$g1" "$g2" 2028600001 2033600001)
      c2ids+=("$id")
      "$PY" - "$H/jobs.json" "c2_worker$index" "$id" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); jobs=json.loads(p.read_text()); jobs[sys.argv[2]]=sys.argv[3].split(';')[0]
p.write_text(json.dumps(jobs,indent=2))
PY
      sbatch --parsable --dependency="after:$id+2" --cpus-per-task=1 --mem=2G --time=00:10:00 \
        --job-name=signsym500k-check --output="$H/logs/check_%j.out" "$H/pipeline.sh" check
    done
    dep=$(IFS=:; echo "${c2ids[*]}")
    final=$(sbatch --parsable --dependency="afterok:$dep" --kill-on-invalid-dep=yes \
      --job-name=signsym500k-final --output="$H/logs/final_%j.out" "$H/pipeline.sh" finish)
    "$PY" - "$H/jobs.json" "$final" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); jobs=json.loads(p.read_text()); jobs['final']=sys.argv[2].split(';')[0]
p.write_text(json.dumps(jobs,indent=2))
PY
    ;;
  finish)
    "$PY" -I -u "$H/study.py" c2
    "$PY" -u "$H/study.py" check
    ;;
  *) echo "unknown stage: $1" >&2; exit 2 ;;
esac
