#!/bin/bash
#SBATCH --job-name=tgt_domB6
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/tgt_domB6_%j.out
set -o pipefail

# PILOT TARGET: the fiducial dom6x6 pin with ONE lever moved -- the SIZE edges.
#
# WHY. Flow #1 over-smooths the steep response transition at Re 0.30-0.42: inside the fiducial pin's
# first size cell the truth rises 0.244 -> 0.586 (2.4x) while the pin supplies a single value, and
# the flow's residual ramps +17% -> -5% across that cell (-24.9 +- 6.6 %/cell) where a pin-free
# control on the same rows is flat (WORKLOG 2026-08-03l, job 15498469).
#
# The size edges here are arm B from scripts/make_response_edges.py -- equal cumulative |d log R|
# with a reported count-floor blend (alpha 0.40) -- chosen on the half-shear ruler with no m in
# evidence. They put 4 of 6 bins inside 0.30-0.42, across the transition.
#
# ONE LEVER ONLY. --n-flux 6 with no --flux-edges reproduces the fiducial equal-count flux edges
# EXACTLY (asserted below), and --n-crowd 5 is unchanged. The response-driven FLUX edges are
# deliberately NOT used: on the magnitude conditional they scored slightly WORSE than equal-count
# (rms 6.78% vs 6.32%, job 15489783), so folding them in would confound the test.
#
# FIREWALL: half-shear g=0.05 leg + its g=0 SNC lookup. constgold not read; no m selected anything.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
R=/home/z/Zekang.Zhang/SBSI/results
OUT=$R/response_target_crowd_rblend_snc_c0-99_6xB6x5_dom.npz
SIZE_EDGES=0.299999,0.316804,0.330889,0.358446,0.420956,0.618024,1.500001

if [ -f "$OUT" ]; then echo "REFUSING to overwrite existing $OUT"; exit 1; fi
echo "### RESP TARGET domB6 job=$SLURM_JOB_ID ###"; date
python -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend \
  --n-flux 6 --n-crowd 5 --min-count 500 --max-case 99 \
  --size-edges "$SIZE_EDGES" \
  --primary-mag-max 26.0 --primary-re-min 0.3 \
  --snc-lookup $R/g0_lookup_c0-99.feather \
  --output "$OUT" 2>&1 | grep -v --line-buffered "module command" || exit 1

echo; echo "### ONE-LEVER ASSERTIONS ###"
python - "$OUT" <<'PY'
import sys, numpy as np
new = np.load(sys.argv[1], allow_pickle=True)
old = np.load("/home/z/Zekang.Zhang/SBSI/results/"
              "response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz", allow_pickle=True)
d = float(np.max(np.abs(new["edges_flux"] - old["edges_flux"])))
print(f"flux edges vs fiducial: max |diff| = {d:.2e}")
assert d < 1e-9, "FLUX EDGES MOVED -- this is no longer a one-lever change"
dc = float(np.max(np.abs(new["edges_crowd"] - old["edges_crowd"])))
print(f"crowd edges vs fiducial: max |diff| = {dc:.2e}")
assert dc < 1e-9, "CROWD EDGES MOVED -- this is no longer a one-lever change"
print("size edges NEW:", np.round(new["edges_size"], 4).tolist())
print("size edges OLD:", np.round(old["edges_size"], 4).tolist())
c = np.asarray(new["counts"], float)
print(f"cells: {c.size}, zero-count {int((c==0).sum())}, min N_eff {int(c.min()):,} "
      f"(fiducial floor 1,875; the 8x8 grid was rejected at 611)")
assert int(c.min()) > 1875, "min cell below the fiducial floor -- thin-cell risk"
assert new["edges_size"][0] <= 0.3 + 1e-3, "size grid must START at the 0.3 cut"
assert new["edges_flux"][-1] >= 26.0 - 1e-3, "flux grid must END at the 26.0 cut"
print("OK: one lever moved, grid aligned to the domain boundary, cells above the floor")
PY
echo "TGT_DOMB6_DONE $OUT"; date
