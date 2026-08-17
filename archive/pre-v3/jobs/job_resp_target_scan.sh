#!/bin/bash
#SBATCH --job-name=tgt_scan
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/tgt_scan_%j.out
set -o pipefail

# The arm-B size edges give a 3-D min cell of 956, BELOW the fiducial's 1,875 floor and near the 611
# that got the 8x8 grid rejected (job 15499440's assertion caught it). The --min-frac guard in
# make_response_edges.py operates on the 1-D MARGINAL, but the binding constraint is the 3-D CELL,
# and the axes are correlated -- small galaxies concentrate in particular mag/crowd cells, so the
# marginal estimate was ~10x optimistic.
#
# Scan candidate size-edge sets and report the resulting 3-D min cell. Pick the most aggressive set
# (lowest alpha = closest to pure equal-d(logR)) that still clears 1,875. Chosen on cell occupancy
# alone -- a statistical property of the grid, with no m and no constgold involved.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
R=/home/z/Zekang.Zhang/SBSI/results
OUTD=/project/ls-gruen/users/zekang.zhang/sbsi_caches/pin_realloc

echo "### TARGET CELL-OCCUPANCY SCAN job=$SLURM_JOB_ID ###"; date
run () {  # $1=label $2=alpha $3=edges
  local OUT=$OUTD/tgt_scan_$1.npz
  echo; echo "===== $1 (alpha=$2) ====="
  python -u scripts/compute_response_target_blend.py \
    --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
    --target-cols measured_ngmix_g1 measured_ngmix_g2 \
    --nominal-g 0.05 --crowd-col r_blend --n-flux 6 --n-crowd 5 \
    --min-count 500 --max-case 99 --size-edges "$3" \
    --primary-mag-max 26.0 --primary-re-min 0.3 \
    --snc-lookup $R/g0_lookup_c0-99.feather --output "$OUT" \
    2>&1 | grep -v --line-buffered "module command" | tail -3 || exit 1
  python - "$OUT" "$1" "$2" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
c = np.asarray(z["counts"], float)
print(f"  RESULT {sys.argv[2]}: alpha={sys.argv[3]}  min 3-D cell N_eff = {int(c.min()):,}  "
      f"{'PASS (>=1875)' if c.min() >= 1875 else 'FAIL (<1875)'}")
print(f"  size edges: {np.round(z['edges_size'],4).tolist()}")
PY
}
run mf06 0.60 0.299999,0.320343,0.345593,0.393329,0.498560,0.720065,1.500001
run mf08 0.75 0.299999,0.325905,0.365498,0.434787,0.551335,0.775155,1.500001
run mf10 0.85 0.299999,0.333741,0.384898,0.462726,0.584669,0.808229,1.500001
echo; echo "reference: fiducial 6x6x5_dom min cell = 1,875 ; arm B (alpha 0.40) = 956"
echo TGT_SCAN_DONE; date
