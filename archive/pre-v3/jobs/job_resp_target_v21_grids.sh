#!/bin/bash
#SBATCH --job-name=resptgt_v21g
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/resptgt_v21g_%j.out
set -eo pipefail

# GRID CHOICE for the V2.1 response target. The 6x6x5 grid copied over from V2 (job 15519854) lands
# at min-cell N_eff = 521. That passes the --min-count 500 floor but FAILS the standard actually
# used for the fiducial: jobs/job_resp_target_domB6.sh asserts min N_eff > 1875 and records that
# "the 8x8 grid was rejected at 611". 521 is below the number that already got a grid rejected.
#
# The cause is not the grid, it is the population: V2.1 keeps 47.5% of V2, so the same 180 cells
# hold half the rows. The pin is a per-cell MEAN response; thin cells make it noisy, and the flow
# is supervised against it.
#
# So build the coarser candidates and report min-cell occupancy for each. Selection is on CELL
# OCCUPANCY and on whether the response structure survives the coarsening -- NOT on constgold m,
# which is evaluation-only (the R_blend firewall).
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
R=/home/z/Zekang.Zhang/SBSI/results
echo "### V2.1 RESP TARGET GRID SCAN job=$SLURM_JOB_ID ###"; date

build () {   # $1=n_flux $2=n_size
  OUT=$R/response_target_crowd_rblend_snc_c0-99_${1}x${2}x5_v21.npz
  echo; echo "--- ${1}x${2}x5 -> $OUT ---"
  python -u scripts/compute_response_target_blend.py \
    --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
    --target-cols measured_ngmix_g1 measured_ngmix_g2 \
    --nominal-g 0.05 --crowd-col r_blend \
    --n-flux "$1" --n-size "$2" --n-crowd 5 --min-count 500 --max-case 99 \
    --v21-domain \
    --snc-lookup $R/g0_lookup_c0-99.feather \
    --output "$OUT" 2>&1 | grep -v --line-buffered "module command" | tail -12
  python - "$OUT" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
c = np.asarray(z["counts"], float); R = z["Rsim"]
w = c.sum(axis=(0, 2)); Rs = (R * c).sum(axis=(0, 2)) / np.where(w > 0, w, np.nan)
print(f"  cells={c.size}  empty={int((c==0).sum())}  min N_eff/cell={int(c.min()):,}  "
      f"median={int(np.median(c)):,}")
print(f"  per-size-bin Rsim: {np.round(Rs, 4).tolist()}")
print(f"  Rsim spread across ALL cells: {np.nanmin(R):.3f}..{np.nanmax(R):.3f}")
print("  VERDICT: " + ("PASSES the fiducial floor (>1875)" if c.min() > 1875
                       else "below the fiducial floor of 1875"))
PY
}

build 6 4
build 5 4
build 4 4
build 5 3
echo "RESPTGT_V21_GRIDS_DONE"; date
