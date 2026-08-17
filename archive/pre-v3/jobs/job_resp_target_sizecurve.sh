#!/bin/bash
#SBATCH --job-name=rtsizecrv
#SBATCH --time=02:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rtsizecrv_%j.out
# INDEPENDENT CONFIRMATION of the size tilt (WORKLOG 2026-08-05z). The tilt was measured as
# R_flow - (R_sim - R_blend) on constgold, so the "demand" it is measured against is model-subtracted
# (CONVENTIONS.md 6b, and the surviving half of the 05n objection). If the demand curve itself is
# wrong in shape, the tilt is an artifact of the instrument rather than a property of the flow.
#
# This builds the demand curve a SECOND way, from HALF-SHEAR only, with fine TRUE-SIZE bins over the
# flow's whole training box. If the half-shear target reproduces the SHAPE of the constgold demand
# across size, the tilt is confirmed by two independent instruments.
#
# WHY A SHAPE COMPARISON IS LEGITIMATE ACROSS CONVENTIONS. The target is forward at |g| = 0.05 and
# constgold is antithetic, which CONVENTIONS.md 6c warns against mixing. 05u measured that term
# directly: at |g| = 0.05 it is -3.41% +- 3.82, consistent with zero, AND it showed no resolution
# trend (V2.1 -0.33%, complement -0.71%, fine Re bins all under 2 sigma). A size-independent offset
# cannot manufacture a SLOPE across size bins. So the absolute levels stay non-comparable and only
# the shape is read here -- that is the whole design of this job.
#
# FIREWALL: the target is built from HALF-SHEAR only. Constgold values enter the printout as a
# labelled COMPARISON, never into the build, and nothing is fitted or corrected.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
R=/home/z/Zekang.Zhang/SBSI/results
OUT=$R/response_target_sizecurve_c0-99_trainbox.npz
set -e
# Fine size edges = the residual map's bins. Flux/crowd conditioning is coarsened (4x3 rather than
# 6x5) so that 11 size bins still leave every cell above --min-count; the size marginal is what is
# being read, and the flux/crowd axes are only there to carry the estimator's own conditioning.
python -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --crowd-col r_blend \
  --size-edges 0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80,1.00,1.20,1.50 \
  --n-flux 4 --n-crowd 3 --min-count 500 --max-case 99 \
  --primary-mag-max 26.0 --primary-re-min 0.3 \
  --snc-lookup $R/g0_lookup_c0-99.feather \
  --output "$OUT" 2>&1 | grep -v --line-buffered "module command" | tail -10
python -u scripts/compare_size_demand.py --npz "$OUT"
echo RTSIZECRV_DONE
