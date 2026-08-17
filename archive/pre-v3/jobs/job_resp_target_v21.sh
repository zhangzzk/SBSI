#!/bin/bash
#SBATCH --job-name=resptgt_v21
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/resptgt_v21_%j.out
set -eo pipefail

# STEP 2 of V2.1: the response target, rebuilt on the V2.1 domain.
#
# ONE LEVER vs jobs/job_resp_target_domain_fine.sh, which built the fiducial
# response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz: the two V2 box flags
# (--primary-mag-max 26.0 --primary-re-min 0.3) are replaced by --v21-domain. Same catalogue, same
# 6x6x5 grid, same estimator, same --min-count, same --max-case.
#
# WHY IT MUST BE REBUILT AT ALL: the target is a MEAN response per (flux, size, crowding) cell,
# measured on a population. Training a V2.1 flow against a V2 target would pin the kept rows to a
# mean that includes rows the trainer never sees -- WORKLOG 2026-07-27d, which cost 5% of R_flow.
#
# CELL OCCUPANCY: V2.1 keeps 47.5% of V2 (job 15519546), so ~2.8M of the ~5.8M in-domain rows,
# ~15k per cell on the 180-cell grid -- still ~30x the --min-count 500 floor. The assertions below
# check that rather than assuming it.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
R=/home/z/Zekang.Zhang/SBSI/results
OUT=$R/response_target_crowd_rblend_snc_c0-99_6x6x5_v21.npz
echo "### V2.1 RESP TARGET job=$SLURM_JOB_ID ###"; date

python -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend \
  --n-flux 6 --n-size 6 --n-crowd 5 --min-count 500 --max-case 99 \
  --v21-domain \
  --snc-lookup $R/g0_lookup_c0-99.feather \
  --output "$OUT" 2>&1 | grep -v --line-buffered "module command" || exit 1

echo; echo "### GRID CHECKS ###"
python - "$OUT" <<'PY'
import sys, numpy as np
sys.path.insert(0, "/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot")
from sbs_shear import domain as D

z = np.load(sys.argv[1], allow_pickle=True)
ef, es, c = z["edges_flux"], z["edges_size"], np.asarray(z["counts"], float)
print("domain stamped in the npz:", str(z["domain"]))
print("edges_flux:", np.round(ef, 4))
print("edges_size:", np.round(es, 4))
print(f"cells={c.size}  empty={int((c == 0).sum())}  min N_eff/cell={int(c.min()):,}")

# The grid must not straddle the domain boundary: a cell half inside the cut averages rows the
# trainer never sees into the pin the trainer is supervised by.
assert es[0] >= D.V21_RE_MIN - 1e-3, f"size grid starts at {es[0]}, below the {D.V21_RE_MIN} cut"
# The S/N cut is a CURVE, so there is no single magnitude edge to assert. What IS fixed is its
# loosest point -- the faintest magnitude anywhere in the domain, reached at the smallest size.
mag_faintest = float(D.sn_limiting_mag(D.V21_RE_MIN))
print(f"faintest magnitude anywhere in the V2.1 domain (at Re={D.V21_RE_MIN}\"): {mag_faintest:.3f}")
assert ef[-1] <= mag_faintest + 0.05, (
    f"flux grid extends to {ef[-1]:.3f}, past the domain's faintest possible {mag_faintest:.3f} "
    f"-- the S/N cut was not applied")
assert int((c == 0).sum()) == 0, "empty cells fall back to a global fill -- reject this grid"
assert int(c.min()) > 500, "a cell is below the --min-count floor"

w = c.sum(axis=(0, 2)); Rs = (z["Rsim"] * c).sum(axis=(0, 2)) / np.where(w > 0, w, np.nan)
for i in range(len(es) - 1):
    print(f"  size bin {i} [{es[i]:.3f},{es[i+1]:.3f}]: Rsim={Rs[i]:+.4f}  N={int(w[i]):,}")
print("OK: grid inside the V2.1 domain, no empty cells, all cells above the floor")
PY
echo "RESPTGT_V21_DONE $OUT"; date
