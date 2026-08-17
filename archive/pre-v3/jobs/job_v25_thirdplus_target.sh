#!/bin/bash
#SBATCH --job-name=v25tptarget
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v25tptarget_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_thirdplus_g0.05_val_full.feather
SNC=/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather
OUT=results/response_target_crowd_thirdplus_snc_c0-99_6x6x5cond_v25.npz
for f in "$CAT" "$SNC"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }

echo "### V2.5: response grid uses the intrinsic third-plus feature directly ###"; date
"$PY" -u scripts/compute_response_target_blend.py \
  --catalogue "$CAT" --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col nbr_flux_thirdplus --n-crowd 5 --crowd-conditional \
  --n-flux 6 --n-size 6 --min-count 500 --max-case 99 \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --snc-lookup "$SNC" --output "$OUT"

"$PY" - "$OUT" <<'PY'
import sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
assert z["Rsim"].shape == (6, 6, 5)
assert str(z["crowd_col"]) == "nbr_flux_thirdplus"
assert bool(z["crowd_conditional"])
assert float(z["primary_mag_max"]) == 25.8
assert float(z["primary_re_min"]) == 0.5
c, r = z["counts"], z["Rsim"]
assert np.all(c > 500), f"response cells below floor: min={c.min()}"
means = [(r[..., q] * c[..., q]).sum() / c[..., q].sum() for q in range(5)]
print(f"cells={c.size}; min/median/max N_eff={c.min():.0f}/{np.median(c):.0f}/{c.max():.0f}")
print("count-weighted response by intrinsic third-plus quintile:", np.round(means, 6))
print("V25_TARGET_ASSERTIONS_PASS")
PY
echo V25_THIRDPLUS_TARGET_DONE; date
