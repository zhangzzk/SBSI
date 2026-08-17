#!/bin/bash
#SBATCH --job-name=cgind_crowd
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cgind_crowd_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/cgind_crowd_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant_indomain
OUT=$ROOT/results/crowd_flux_const_indomain_c40-89.feather
CASES=({40..89})
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }
"$PY" -u scripts/build_crowding_lookup.py --cases "${CASES[@]}" \
  --base "$BASE" --sign 0.02 --output "$OUT"
echo CONSTGOLD_INDOMAIN_CROWD_LOOKUP_DONE
