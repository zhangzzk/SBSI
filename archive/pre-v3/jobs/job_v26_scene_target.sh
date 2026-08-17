#!/bin/bash
#SBATCH --job-name=v26target
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v26target_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_scene_g0.05_val_full.feather
SNC=/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather
OUT=results/response_target_scene_top2_thirdplus_snc_c0-99_6x6x5x4cond_v26.npz
for f in "$CAT" "$SNC"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }

echo "### V2.6: joint intrinsic top-two x third-plus response grid ###"; date
"$PY" -u scripts/compute_response_target_blend.py \
  --catalogue "$CAT" --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 \
  --crowd-col nbr_flux_top2 --n-crowd 5 --crowd-conditional \
  --crowd2-col nbr_flux_thirdplus --n-crowd2 4 --crowd2-conditional \
  --n-flux 6 --n-size 6 --min-count 500 --max-case 99 \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --snc-lookup "$SNC" --output "$OUT"

"$PY" - "$OUT" <<'PY'
import sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
assert z["Rsim"].shape == (6, 6, 5, 4)
assert str(z["crowd_col"]) == "nbr_flux_top2"
assert str(z["crowd2_col"]) == "nbr_flux_thirdplus"
assert bool(z["crowd_conditional"]) and bool(z["crowd2_conditional"])
assert float(z["primary_mag_max"]) == 25.8
assert float(z["primary_re_min"]) == 0.5
assert abs(float(z["global_R"]) - 0.815667) < 5e-6, z["global_R"]
c, r = z["counts"], z["Rsim"]
assert np.all(c > 500), f"response cells below floor: min={c.min()}"
means = [(r[..., q] * c[..., q]).sum() / c[..., q].sum() for q in range(4)]
print(f"cells={c.size}; min/median/max N_eff={c.min():.0f}/{np.median(c):.0f}/{c.max():.0f}")
print("count-weighted response by conditional third-plus quartile:", np.round(means, 6))
print("V26_SCENE_TARGET_ASSERTIONS_PASS")
PY
echo V26_SCENE_TARGET_DONE; date
