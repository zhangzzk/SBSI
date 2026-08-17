#!/bin/bash
#SBATCH --job-name=rcurv02
#SBATCH --time=4:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rcurv02_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/rcurv02_%j.err

# THE CURVATURE TEST THAT ACTUALLY HAS A LEVER ARM (WORKLOG 2026-08-03d).
#
# Job 15485364 tried to constrain `c` in R(g) = R_0 + c g^2 from g=0.02 vs g=0.05. It returned a
# clean NULL -- blend c = -2.97 +- 3.22, self c = +0.67 +- 1.58 -- but the bound is USELESS at the
# amplitude we care about: the 2-sigma limit |c| < 6.44 allows up to 327% contamination of the blend
# response at g=0.2. The lever arm 0.05^2 - 0.02^2 = 0.0021 is simply too short.
#
# The g=0.2 pair set (stage 3) fixes that for free:
#     0.02 vs 0.2  ->  lever arm 0.0396  (19x longer)
#     0.05 vs 0.2  ->  lever arm 0.0375  (18x longer)
# so the SAME script, on the SAME labels, becomes ~18x more constraining. This is a DIRECT measurement
# at the amplitude in question rather than an extrapolation to it.
#
# READ IT THIS WAY. If R(0.2) sits on the R_0 implied by the low-g pair, curvature is negligible and
# the g=0.2 retrain is a clean 4x noise reduction. If it does not, the g=0.2 LABELS are biased, the
# 16-seed retrain running alongside inherits that bias, and the right move is g=0.05 with more seeds
# instead of more shear. Either way this must be read BEFORE any g=0.2 `m` is quoted.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow

for f in blend_pairset_ap7_g002.feather blend_pairset_ap7_dual.feather blend_pairset_ap7_g02.feather; do
  [ -f "$CACHE/$f" ] || { echo "REFUSING: missing $CACHE/$f"; exit 1; }
done

echo "### LONGEST LEVER ARM: g=0.02 vs g=0.2 (0.0396)"; date
python -u scripts/measure_response_curvature.py \
    --pairset "$CACHE/blend_pairset_ap7_g002.feather" "$CACHE/blend_pairset_ap7_g02.feather" \
    --shear 0.02 0.2 || exit 1

echo; echo "### AGAINST THE CURRENT TRAINING AMPLITUDE: g=0.05 vs g=0.2 (0.0375)"; date
python -u scripts/measure_response_curvature.py \
    --pairset "$CACHE/blend_pairset_ap7_dual.feather" "$CACHE/blend_pairset_ap7_g02.feather" \
    --shear 0.05 0.2 || exit 1

echo "RCURV02_DONE"; date
