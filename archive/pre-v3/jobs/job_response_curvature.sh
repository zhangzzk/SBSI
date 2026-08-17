#!/bin/bash
#SBATCH --job-name=rcurv
#SBATCH --time=8:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rcurv_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/rcurv_%j.err

# HOW BIG IS THE g^2 CURVATURE IN THE RESPONSE LABEL? (WORKLOG 2026-08-03b open item.)
#
# The owner has approved retraining flow #2 at g = 0.2. That buys 4x less label noise and pays 16x
# the curvature term, since R(g) = R_0 + c g^2. I flagged that `c` is unmeasured; this measures it,
# using ONLY legs that already exist, so the answer lands before the g=0.2 retrain does.
#
# Builds the pair set at g = 0.02 with the SAME builder and the SAME g=0 reference leg as the
# existing g = 0.05 pair set -- so the two differ in shear amplitude and nothing else -- then solves
# R_0 and c from the pair.
#
# Uses the g0.02 test100 leg (66 GB, 100 cases) rather than the 13 GB test leg, for statistics: the
# lever arm 0.05^2 - 0.02^2 = 0.0021 amplifies the response difference by ~476x into `c`, so this
# measurement is only as good as the precision on that difference.
#
# FIREWALL: half-shear legs only, constgold never opened.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow
PS002=$CACHE/blend_pairset_ap7_g002.feather
PS005=$CACHE/blend_pairset_ap7_dual.feather

if [ -f "$PS002" ]; then
  echo "### g=0.02 pair set already exists, skipping build"; ls -la "$PS002"
else
  echo "### BUILD g=0.02 pair set (same builder, same g=0 reference leg)"; date
  python -u scripts/build_blend_pairset.py \
      --gs-leg "$CAT/det_meas_ngmix_ap7_g0.02_test100.feather" \
      --g0-leg "$CAT/det_meas_ngmix_ap7_g0.0_train.feather" \
      --output "$PS002" || exit 1
fi

echo; echo "### CURVATURE: solve R_0 and c from g=0.02 vs g=0.05"; date
python -u scripts/measure_response_curvature.py \
    --pairset "$PS002" "$PS005" --shear 0.02 0.05 || exit 1

echo "RCURV_DONE"; date
