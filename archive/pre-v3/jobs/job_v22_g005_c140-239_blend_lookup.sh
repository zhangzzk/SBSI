#!/bin/bash
#SBATCH --job-name=v22g5_blend
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22g5_blend_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant_g005_c140-239
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_g005_c140-239
OUT="$RUN/blend_lookup_v22_c140-239.feather"
mkdir -p "$RUN"
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
for case in $(seq 140 239); do
  input="$BASE/case${case}_0.05/real0/catalogues/input/gals_info_tile180.0_-0.5.feather"
  [ -f "$input" ] || { echo "MISSING $input"; exit 1; }
done

echo "### FROZEN V2.2 BLENDEMU LOOKUP: g=0.05 cases 140--239, job=$SLURM_JOB_ID ###"
date
"$PY" -u scripts/build_blend_lookup.py \
  --cases $(seq 140 239) --base "$BASE" --sign 0.05 \
  --tag lsst_r_extnbr_v22 --output "$OUT"
"$PY" - "$OUT" <<'PY'
import sys
import pyarrow.feather as pf

path = sys.argv[1]
t = pf.read_table(path, columns=["case", "input_index", "R_blend"])
cases = set(t["case"].to_numpy().tolist())
assert cases == set(range(140, 240)), (min(cases), max(cases), len(cases))
assert t.num_rows > 0
print(f"validated {path}: {t.num_rows:,} rows, 100 cases")
PY
echo V22_G005_BLEND_LOOKUP_DONE
date
