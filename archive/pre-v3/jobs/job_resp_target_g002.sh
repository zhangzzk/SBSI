#!/bin/bash
#SBATCH --job-name=resptgt002
#SBATCH --time=03:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/resptgt002_%j.out
set -eo pipefail

# THE TARGET-DEFINITION TEST, firewall-clean.
#
# The pin target is built with the SNC estimator, R_snc(g) = <[e(g) - e(0)].ghat>/g, on the g=0.05
# half-shear leg. That is a FORWARD difference, so it carries an O(g) curvature term:
#       R_snc(g) = R_0 + c*g + O(g^2)
# The acceptance metric's r_sim on constgold is ANTITHETIC, <(e_+g - e_-g).ghat>/(2g), which cancels
# the even terms and measures R_0 directly. So the target and the metric are estimating DIFFERENT
# quantities, and the difference is c*0.05 -- a systematic the flow cannot train away. My own earlier
# note records exactly this class of effect at ~20% (+0.49 vs +0.60 on a faint self-response,
# antithetic vs forward, on image-identical sims).
#
# The builder HAS an --antithetic mode, but its help says it reads "a constant-shear ANTITHETIC
# (+g/-g) render catalogue ... the SAME sample as the acceptance metric's r_sim", i.e. CONSTGOLD.
# Building a training target that way trains the flow on the acceptance truth and is a FIREWALL
# VIOLATION. It is deliberately NOT used here.
#
# The clean route uses only half-shear legs: build the SAME target on the g=0.02 leg. Comparing
# R_snc(0.02) with R_snc(0.05) measures c, and (5*R(0.02) - 2*R(0.05))/3 Richardson-extrapolates to
# g->0 -- the quantity the metric actually needs, obtained without ever reading constgold.
#
# Everything else is byte-identical to jobs/job_resp_target_domain.sh at 6x6x5: same grid sizes, same
# crowd column, same snc lookup, same domain cuts, same max-case.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
OUT=/home/z/Zekang.Zhang/SBSI/results/response_target_crowd_rblend_snc_g002_c0-99_6x6x5_dom.npz

echo "### RESP TARGET g=0.02 job=$SLURM_JOB_ID ###"; date
python -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_crowd_g0.02_test_full.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.02 --crowd-col r_blend \
  --n-flux 6 --n-size 6 --n-crowd 5 --min-count 500 --max-case 99 \
  --primary-mag-max 26.0 --primary-re-min 0.3 \
  --snc-lookup /home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather \
  --output $OUT 2>&1 | grep -v "module command"

echo; echo "### g=0.02 vs g=0.05: is the forward-difference curvature real? ###"
python - "$OUT" <<'PY'
import sys, numpy as np
new = np.load(sys.argv[1], allow_pickle=True)
old = np.load("/home/z/Zekang.Zhang/SBSI/results/"
              "response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz", allow_pickle=True)
for k in ("edges_flux", "edges_size", "edges_crowd"):
    d = np.max(np.abs(new[k] - old[k]))
    print(f"{k}: max edge difference {d:.5f}  {'OK - cell-by-cell comparison valid' if d < 5e-3 else '*** EDGES DIFFER: cells are not comparable ***'}")
Rn, Ro = new["Rsim"], old["Rsim"]
cn, co = np.asarray(new["counts"], float), np.asarray(old["counts"], float)
wn = cn / cn.sum(); wo = co / co.sum()
print(f"\ncount-weighted mean R:  g=0.02 -> {(Rn*wn).sum():.4f}    g=0.05 -> {(Ro*wo).sum():.4f}")
print(f"  difference R(0.02) - R(0.05) = {(Rn*wn).sum() - (Ro*wo).sum():+.4f}")
# Richardson to g->0 assuming R_snc(g) = R0 + c*g
R0 = (5.0 * Rn - 2.0 * Ro) / 3.0
print(f"  extrapolated to g->0        = {(R0*wn).sum():.4f}")
print(f"\nfor reference, m=0 on the in-domain constgold population needs <R_flow> = 0.7236")
print(f"(measured: <r_sim>=0.8607, <R_blend>=0.1371, dom6x6 achieves <R_flow>=0.7280)")
print("\nper crowd bin (count-weighted over mag,size):")
for c in range(Rn.shape[2]):
    a = (Rn[:, :, c]*cn[:, :, c]).sum()/cn[:, :, c].sum()
    b = (Ro[:, :, c]*co[:, :, c]).sum()/co[:, :, c].sum()
    e = (R0[:, :, c]*cn[:, :, c]).sum()/cn[:, :, c].sum()
    print(f"  q{c}: g=0.02 {a:.4f}   g=0.05 {b:.4f}   diff {a-b:+.4f} ({100*(a-b)/b:+.1f}%)   g->0 {e:.4f}")
print("\nper size bin:")
for s in range(Rn.shape[1]):
    a = (Rn[:, s, :]*cn[:, s, :]).sum()/cn[:, s, :].sum()
    b = (Ro[:, s, :]*co[:, s, :]).sum()/co[:, s, :].sum()
    print(f"  size {s} [{new['edges_size'][s]:.3f},{new['edges_size'][s+1]:.3f}]: "
          f"g=0.02 {a:.4f}   g=0.05 {b:.4f}   diff {a-b:+.4f} ({100*(a-b)/b:+.1f}%)")
PY
echo "RESPTGT002_DONE $OUT"; date
