#!/bin/bash
#SBATCH --job-name=resptgt_lows
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/resptgt_lows_%j.out
set -eo pipefail

# BOUNDARY-REFINED size grid. Motivated by measurement, not by a constgold number: on the half-shear
# ruler the dom6x6 ensemble's response error is concentrated in true size [0.30,0.38) -- -3.14% there
# (2.2 sigma vs truth noise) against <1 sigma in all three larger bins. Weighted by share, that single
# bin contributes -0.53% of the -0.35% overall gap. Every other bin is consistent with zero.
#
# The 6x6 quantile edges put only ONE boundary (0.355) inside that region, so the flow interpolates
# across the steep small-size response step (the documented limiter, WORKLOG cont.110f). This splits
# the bottom two quantile bins into five, leaving the upper edges untouched so the change is isolated.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
R=/home/z/Zekang.Zhang/SBSI/results
OUT=$R/response_target_crowd_rblend_snc_c0-99_6x9lows5_dom.npz
echo "### LOW-SIZE REFINED RESP TARGET job=$SLURM_JOB_ID ###"; date

python -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend \
  --n-flux 6 --n-crowd 5 --min-count 500 --max-case 99 \
  --size-edges "0.300,0.318,0.336,0.355,0.387,0.419,0.503,0.629,0.853,1.500" \
  --primary-mag-max 26.0 --primary-re-min 0.3 \
  --snc-lookup $R/g0_lookup_c0-99.feather \
  --output "$OUT" 2>&1 | grep -v "module command"

python - "$OUT" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1]); c = z["counts"]; es, ef = z["edges_size"], z["edges_flux"]
print("  edges_size:", np.round(es, 4))
print(f"  cells={c.size}  empty={int((c==0).sum())}  min/cell={int(c.min()):,}")
w = c.sum(axis=(0, 2)); R = z["Rsim"]
Rs = (R * c).sum(axis=(0, 2)) / np.where(w > 0, w, np.nan)
print("  per-size-bin N   :", [f"{int(x):,}" for x in w])
print("  per-size-bin Rsim:", np.round(Rs, 4))
assert es[0] <= 0.3 + 1e-3 and ef[-1] >= 26.0 - 1e-3, "grid must align to the domain boundary"
assert int((c == 0).sum()) == 0, "empty cells fall back to a global fill -- reject this grid"
print("  OK aligned, no empty cells")
PY
echo "RESPTGT_LOWS_DONE"; date
