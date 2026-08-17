#!/bin/bash
#SBATCH --job-name=rt21c4099
#SBATCH --time=00:40:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rt21c4099_%j.out

# POPULATION TEST for the 2.53% response-target gap (WORKLOG 2026-08-04p candidate 3).
#
# The V2.1 response target is built on half-shear cases 0-99; constgold's m is measured on cases
# 40-139. If part of the gap is the case window rather than the estimator, rebuilding the target on
# the 40-99 OVERLAP will move it. If the target is unchanged, the population is exonerated and the
# remaining candidates are the snc g=0 reference and the emulator.
#
# ONE LEVER: --min-case 40 added to the exact command of jobs/job_resp_target_v21_grids.sh. Same
# catalogue, same snc lookup, same 5x4x5 grid, same V2.1 domain, same nominal g.
#
# EXPECTED, WRITTEN BEFORE THE RUN: cases are independent renders of the same population, so the
# target should be UNCHANGED within its sampling error (+-0.0014, i.e. +-0.16%, measured
# 2026-08-04r). A shift beyond ~0.3% would mean the case windows are not statistically equivalent,
# which would be a finding in its own right and would cast doubt on every case-window comparison in
# this project. A null here is the boring and expected outcome.
#
# NOTE the grid edges are QUANTILES of the kept rows, so they move slightly with the case window;
# global_R (the population-weighted mean, which is the number being tested) does not depend on the
# binning at all, so read that and not the per-cell values.
#
# FIREWALL: half-shear only; constgold is never opened.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
R=/home/z/Zekang.Zhang/SBSI/results
OUT=$R/response_target_crowd_rblend_snc_c40-99_5x4x5_v21.npz

echo "### RESP TARGET V2.1 cases 40-99 job=$SLURM_JOB_ID ###"; date
python -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend \
  --n-flux 5 --n-size 4 --n-crowd 5 --min-count 500 \
  --min-case 40 --max-case 99 \
  --v21-domain \
  --snc-lookup $R/g0_lookup_c0-99.feather \
  --output "$OUT" 2>&1 | grep -v --line-buffered "module command" || { echo RT_FAILED; exit 1; }

echo; echo "### COMPARISON ###"
python - "$OUT" $R/response_target_crowd_rblend_snc_c0-99_5x4x5_v21.npz <<'PY'
import sys, numpy as np
new, old = (np.load(p, allow_pickle=True) for p in sys.argv[1:3])
a, b = float(new["global_R"]), float(old["global_R"])
print(f"  cases 40-99 : global_R = {a:.5f}   N_eff = {new['counts'].sum():,.0f}")
print(f"  cases  0-99 : global_R = {b:.5f}   N_eff = {old['counts'].sum():,.0f}")
print(f"  shift       = {a-b:+.5f}  ({100*(a-b)/b:+.2f}%)")
print(f"  target sampling error is +-0.0014 (0.16%, WORKLOG 2026-08-04r, cases 0-99).")
print(f"  constgold demands 0.87241 with the CURRENT emulator; gap from cases 40-99 = "
      f"{100*(0.87241-a)/a:+.2f}%")
PY
echo RT_C4099_DONE; date
