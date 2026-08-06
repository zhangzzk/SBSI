#!/bin/bash
#SBATCH --job-name=resptgt_v22
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/resptgt_v22_%j.out
set -euo pipefail

# V2.2 changes only the rectangular intrinsic-primary domain relative to V2. The estimator,
# half-shear catalogue, 6x6x5 grid, SNC construction and case range are unchanged.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
MAIN_RESULTS=/home/z/Zekang.Zhang/SBSI/results
OUT=results/response_target_crowd_rblend_snc_c0-99_6x6x5_v22.npz
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
echo "### V2.2 RESPONSE TARGET: true r<25.8, Re>0.5 arcsec job=$SLURM_JOB_ID ###"; date

python -u scripts/compute_response_target_blend.py \
  --catalogue "$D/det_meas_crowd_g0.05_val_full.feather" \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend \
  --n-flux 6 --n-size 6 --n-crowd 5 --min-count 500 --max-case 99 \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --snc-lookup "$MAIN_RESULTS/g0_lookup_c0-99.feather" \
  --output "$OUT" 2>&1 | grep -v --line-buffered "module command"

python - "$OUT" <<'PY'
import sys
import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)
ef, es, counts = z["edges_flux"], z["edges_size"], z["counts"]
print("domain:", str(z["domain"]))
print("edges_flux:", np.round(ef, 4))
print("edges_size:", np.round(es, 4))
print(f"cells={counts.size} empty={int((counts == 0).sum())} min_N_eff={int(counts.min()):,}")
assert float(z["primary_mag_max"]) == 25.8
assert float(z["primary_re_min"]) == 0.5
assert es[0] <= 0.5 + 1e-3 and ef[-1] >= 25.8 - 1e-3
assert int((counts == 0).sum()) == 0
assert float(counts.min()) > 500
print("OK: exact V2.2 box stamped; all 180 cells occupied above the floor")
PY
echo "RESPTGT_V22_DONE $OUT"; date
