#!/bin/bash
#SBATCH --job-name=resptgt_v21_g002
#SBATCH --time=01:00:00
#SBATCH --mem=16G          # MEASURED: the g=0.05 build (15519919) peaked at 869M in 74s
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/resptgt_v21_g002_%j.out
set -eo pipefail

# IS THE V2.1 RESPONSE TARGET BIASED BY FORWARD-DIFFERENCE TRUNCATION?
#
# The V2.1 flow trains toward a target whose population-weighted self-response is 0.8509, and it is
# ON that target to 0.13% (WORKLOG 2026-08-04j). But constgold DEMANDS a self-response of
# R_sim - R_blend = 0.99027 - 0.11786 = 0.8724 -- 2.53% HIGHER. So the flow is not failing to learn
# its target; the TARGET disagrees with the acceptance metric. Training harder would make m WORSE:
# a flow that hit 0.8509 exactly on constgold would give m = +2.22%, against the +0.790% we have.
#
# PRIME SUSPECT: the two numbers use DIFFERENT RESPONSE ESTIMATORS.
#   target    : FORWARD difference, (e(g) - e_snc(0))/g, at g = 0.05   [`snc` mode]
#   constgold : CENTRAL difference, (e(+g) - e(-g))/(2g), at g = 0.02  [antithetic]
# A forward difference carries an O(g) error from the second derivative; a central difference
# cancels it and is O(g^2). At g = 0.05 that truncation is 2.5x larger than at g = 0.02.
#
# THE TEST: rebuild the SAME target with ONE lever changed -- the g = 0.02 leg instead of g = 0.05.
# Same script, same 5x4x5 grid, same --v21-domain, same snc reference, same --max-case 99. The two
# catalogues are directly comparable: both cases 0-99, 15,704,454 vs 15,697,220 rows (0.05% apart).
#
# FALSIFIABLE PREDICTION. If R_fwd(g) = R0 + a g and the central-difference 0.8724 is ~R0, then
# a = (0.8509 - 0.8724)/0.05 = -0.43, so the g = 0.02 target should land near
#     0.8724 - 0.02*0.43 = 0.864,  i.e. ~1.5% ABOVE the g = 0.05 target's 0.8509.
# Land near 0.864 -> truncation confirmed, and a two-point extrapolation to g -> 0 is the fix.
# Land near 0.851 -> truncation is NOT the mechanism and the gap is sim/population/estimator-
#   reference instead. Either way this is a measurement, not a tuning knob.
#
# FIREWALL: half-shear only. Nothing here reads constgold; the 0.8724 appears only as the number
# being explained. Do NOT rebuild the target to match constgold -- that would be training on the
# acceptance metric. The legitimate fix is a BETTER ESTIMATOR, chosen on estimator theory.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
R=/home/z/Zekang.Zhang/SBSI/results
OUT=$R/response_target_crowd_rblend_snc_c0-99_5x4x5_v21_g002.npz
echo "### V2.1 RESP TARGET @ g=0.02 job=$SLURM_JOB_ID ###"; date

python -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_crowd_g0.02_test_full.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.02 --crowd-col r_blend \
  --n-flux 5 --n-size 4 --n-crowd 5 --min-count 500 --max-case 99 \
  --v21-domain \
  --snc-lookup $R/g0_lookup_c0-99.feather \
  --output "$OUT" 2>&1 | grep -v --line-buffered "module command" || exit 1

echo; echo "### FORWARD-DIFFERENCE TRUNCATION TEST ###"
python - "$OUT" "$R/response_target_crowd_rblend_snc_c0-99_5x4x5_v21.npz" <<'PY'
import sys, numpy as np
new, old = np.load(sys.argv[1], allow_pickle=True), np.load(sys.argv[2], allow_pickle=True)
def wmean(d):
    c = np.asarray(d["counts"], float).reshape(-1)
    return float((np.asarray(d["Rsim"], float).reshape(-1) * c / c.sum()).sum()), int(c.sum())
r2, n2 = wmean(new); r5, n5 = wmean(old)
CG = 0.99027 - 0.11786          # what constgold demands; NOT used to build anything
print(f"  g=0.05 target : {r5:.4f}  (N={n5:,})")
print(f"  g=0.02 target : {r2:.4f}  (N={n2:,})")
print(f"  constgold     : {CG:.4f}  (central difference, g=0.02)")
print(f"  predicted g=0.02 if pure truncation: 0.8638")
# two-point linear extrapolation of the forward difference to g -> 0
r0 = r2 - (r5 - r2) * 0.02 / 0.03
print(f"\n  extrapolated to g -> 0 : {r0:.4f}   (residual vs constgold: {100*(CG/r0-1):+.2f}%)")
print(f"  moved {100*(r2/r5-1):+.2f}% of the {100*(CG/r5-1):+.2f}% gap")
print("\nVERDICT: >~1% of the move recovered -> truncation is real and extrapolation is the fix.")
print("         ~0% -> truncation is NOT the mechanism; look at the snc reference or the sims.")
PY
echo RESPTGT_V21_G002_DONE; date
