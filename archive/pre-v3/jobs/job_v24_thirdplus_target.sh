#!/bin/bash
#SBATCH --job-name=v24tptarget
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v24tptarget_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
LOOKUP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/scene/neighbor_thirdplus_hs_c0-199.feather
CAT=$D/det_meas_crowd_thirdplus_g0.05_val_full.feather
OUT=results/response_target_crowd_rblend_thirdplus_snc_c0-99_6x6x5x2cond_v24.npz
SNC=/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather
for f in "$D/det_meas_crowd_g0.05_val_full.feather" "$LOOKUP" "$SNC"; do
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
done
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }

echo "### V2.4: augment half-shear target catalogue with intrinsic third-plus flux ###"; date
if [ ! -e "$CAT" ]; then
  "$PY" -u scripts/augment_catalogue_lookup.py \
    --catalogue "$D/det_meas_crowd_g0.05_val_full.feather" --lookup "$LOOKUP" \
    --columns nbr_flux_thirdplus --output "$CAT"
else
  "$PY" - "$CAT" <<'PY'
import sys
import pyarrow as pa
import pyarrow.ipc as ipc
with ipc.open_file(pa.memory_map(sys.argv[1])) as reader:
    assert "nbr_flux_thirdplus" in reader.schema.names
    print(f"reusing verified augmented target catalogue ({reader.num_record_batches} batches)")
PY
fi

echo "### V2.4: 4-D response target (V2.2 grid + third-plus scene axis) ###"; date
"$PY" -u scripts/compute_response_target_blend.py \
  --catalogue "$CAT" \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend --n-crowd 5 \
  --crowd2-col nbr_flux_thirdplus --n-crowd2 2 --crowd2-conditional \
  --n-flux 6 --n-size 6 --min-count 500 --max-case 99 \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --snc-lookup "$SNC" --output "$OUT"

"$PY" - "$OUT" <<'PY'
import sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
assert z["Rsim"].shape == (6, 6, 5, 2)
assert str(z["crowd_col"]) == "r_blend"
assert str(z["crowd2_col"]) == "nbr_flux_thirdplus"
assert bool(z["crowd2_conditional"])
assert float(z["primary_mag_max"]) == 25.8
assert float(z["primary_re_min"]) == 0.5
counts = z["counts"]
print(f"cells={counts.size}; below/equal floor={(counts <= 500).sum()}; "
      f"min/median/max N_eff={counts.min():.0f}/{np.median(counts):.0f}/{counts.max():.0f}")
print("V24_TARGET_ASSERTIONS_PASS")
PY
echo V24_THIRDPLUS_TARGET_DONE; date
