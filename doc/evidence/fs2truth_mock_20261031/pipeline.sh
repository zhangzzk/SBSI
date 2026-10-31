#!/bin/bash
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# Test (a) stages. At most 4 GPUs at any time: mock (1) -> preflight (1) -> C1: 2 workers per leg (4) ->
# per leg after its C1: estimate, then 2 C2 workers for that leg -> C2 estimate -> pair when both legs are done.
set -euo pipefail
H=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2truth_mock_signsym_20261031
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
SB=/opt/slurm/bin/sbatch
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset PYTHONPATH
cd "$H"
record() { "$PY" -c "import json,sys;from pathlib import Path;p=Path(sys.argv[1]);j=json.loads(p.read_text()) if p.exists() else {};j[sys.argv[2]]=sys.argv[3].split(';')[0];p.write_text(json.dumps(j,indent=2))" "$H/jobs.json" "$1" "$2"; }
case "$1" in
  legs) "$PY" -u "$H/study.py" legs ;;
  launch-c1)
    for leg in plus minus; do
      read -r seed bank < <("$PY" -c "import json;s=json.load(open('$H/rule.json'))['seeds']['$leg']['c1'];print(*s)")
      ids=()
      for w in 0 1; do
        id=$($SB --parsable --job-name=fs2mock-c1-$leg --output="$H/$leg/logs/c1_%j.out" "$H/$leg/worker.sh" c1 0 0 "$seed" "$bank")
        record "c1_${leg}_worker$w" "$id"; ids+=("$id")
      done
      dep=$(IFS=:; echo "${ids[*]}")
      id=$($SB --parsable --dependency="afterok:$dep" --kill-on-invalid-dep=yes --job-name=fs2mock-c1est-$leg \
        --output="$H/logs/c1done_${leg}_%j.out" "$H/pipeline.sh" c1-done "$leg")
      record "c1_done_$leg" "$id"
    done ;;
  c1-done)
    leg=$2
    "$PY" -I -u "$H/study.py" estimate "$leg" c1
    read -r g1 g2 < <("$PY" -I -c "import json;print(*(round(x,5) for x in json.load(open('$H/$leg/c1/estimate.json'))['g']))")
    read -r seed bank < <("$PY" -c "import json;s=json.load(open('$H/rule.json'))['seeds']['$leg']['c2'];print(*s)")
    ids=()
    for w in 0 1; do
      id=$($SB --parsable --job-name=fs2mock-c2-$leg --output="$H/$leg/logs/c2_%j.out" "$H/$leg/worker.sh" c2 "$g1" "$g2" "$seed" "$bank")
      record "c2_${leg}_worker$w" "$id"; ids+=("$id")
    done
    dep=$(IFS=:; echo "${ids[*]}")
    id=$($SB --parsable --dependency="afterok:$dep" --kill-on-invalid-dep=yes --job-name=fs2mock-c2est-$leg \
      --output="$H/logs/c2done_${leg}_%j.out" "$H/pipeline.sh" c2-done "$leg")
    record "c2_done_$leg" "$id" ;;
  c2-done)
    leg=$2
    "$PY" -I -u "$H/study.py" estimate "$leg" c2
    if [ -s "$H/plus/c2/estimate.json" ] && [ -s "$H/minus/c2/estimate.json" ]; then
      "$PY" -I -u "$H/study.py" pair
    fi ;;
  *) echo "unknown stage: $1" >&2; exit 2 ;;
esac
