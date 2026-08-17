#!/bin/bash
#SBATCH --job-name=resptgt_dom
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/resptgt_dom_%j.out
set -eo pipefail

# Shape-response target rebuilt ON the deliverable domain (primary true mag<26, Re>0.3).
#
# WHY (WORKLOG 2026-07-27d): the first domain-training attempt reused the full-population target,
# whose size grid is [0.1, 0.241, 0.412, 1.5]. Re>0.3 DELETES bin 0 and SLICES bin 1, so the
# surviving Re in [0.30,0.412] rows were pinned to +0.2290 -- a mean measured over the whole
# [0.241,0.412] bin, including the smaller galaxies the trainer no longer sees. Response climbs
# steeply across those bins (-0.087 / +0.229 / +0.702), so the survivors were pinned too LOW,
# R_flow came out 5% low in-domain, and m went +4.75% -> +9.31%.
#
# The fix is minimal on purpose: SAME grid resolution (6 flux x 3 size x 5 crowd), but the edges are
# now quantiles of the DOMAIN population, so no bin straddles the cut. As a side benefit the 3 size
# bins now span [0.3,1.5] instead of [0.1,1.5], i.e. finer resolution inside the domain for free.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
OUT=/home/z/Zekang.Zhang/SBSI/results/response_target_crowd_rblend_snc_c0-99_6x3x5_dom.npz
echo "### DOMAIN RESP TARGET job=$SLURM_JOB_ID ###"; date

python -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend \
  --n-flux 6 --n-size 3 --n-crowd 5 --min-count 500 --max-case 99 \
  --primary-mag-max 26.0 --primary-re-min 0.3 \
  --snc-lookup /home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather \
  --output $OUT 2>&1 | grep -v "module command"

echo; echo "### grid check: no bin may straddle the cut ###"
python - "$OUT" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1])
ef, es, c = z["edges_flux"], z["edges_size"], z["counts"]
print("edges_flux:", np.round(ef, 4))
print("edges_size:", np.round(es, 4))
print(f"cells with counts==0: {int((c == 0).sum())} / {c.size}")
print(f"min per-cell count  : {int(c.min()):,}")
w = c.sum(axis=(0, 2))
R = z["Rsim"]
Rs = (R * c).sum(axis=(0, 2)) / np.where(w > 0, w, np.nan)
for i in range(len(es) - 1):
    print(f"  size bin {i} [{es[i]:.3f},{es[i+1]:.3f}]: Rsim={Rs[i]:+.4f}  N={int(w[i]):,}")
assert es[0] <= 0.3 + 1e-3, "size grid must START at the 0.3 cut"
assert ef[-1] >= 26.0 - 1e-3, "flux grid must END at the 26.0 cut"
print("OK: grid aligned to the domain boundary")
PY
echo "RESPTGT_DOM_DONE $OUT"; date
