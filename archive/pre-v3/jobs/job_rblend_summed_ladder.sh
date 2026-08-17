#!/bin/bash
#SBATCH --job-name=rbsumlad
#SBATCH --time=02:00:00
#SBATCH --mem=160G          # --all-neighbours keeps ~4 rows per primary instead of 1, so this holds
                            # ~4x the per-pair ruler's frame (which measured 79G at ap7 g=0.2).
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rbsumlad_%j.out

# THE MEASUREMENT THAT REMOVES THE TRANSFER ASSUMPTION.
#
# 2026-08-05b established the emulator under-predicts per-PAIR blend response by -6.15% (3.6 sigma)
# but could not convert that into `m`, because `m` uses R_blend SUMMED over a primary's neighbours
# and the pair list alone moves the ratio 2x (ap7 -4.38% vs non-ap7 -8.89% at the same shear).
#
# `--all-neighbours` keeps every neighbour row instead of one per primary, and
# eval_rblend_gap_summed.py sums each neighbour's OWN projection per primary. That works because
# every secondary carries an INDEPENDENT random shear direction (verified: 4.4 distinct angles per
# primary, 0% share one), so each per-pair projection is an unbiased estimator of that neighbour's
# response and the sum estimates sum_j R_blend(j) -- the same quantity build_blend_lookup.py builds.
# No per-pair-to-sum scaling is assumed anywhere.
#
# g=0.2 for precision (4x signal; linearity verified to 0.6 sigma, 2026-08-05b) on the ap7 legs.
# APERTURE CAVEAT, stated up front: ap7 annotates neighbours to 7" while the emulator's native
# lookup call uses r_max=10", k=20. So the sim-side sum is over a 7" aperture and the 7-10" shell is
# missing from it. Compare like with like -- the emulator column here is summed over the SAME rows.
#
# FIREWALL: half-shear legs only; constgold is never opened.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
C=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
E=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval; mkdir -p $E
NPZ=$E/rblend_measured_allnbr_v21dom_ap7g0.2_ladder.npz

echo "### SUMMED R_BLEND V2.1 job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_rblend_gap_measured.py \
  --gs-leg $C/det_meas_ngmix_ap7_g0.2_val.feather \
  --g0-leg $C/det_meas_ngmix_ap7_g0.0_train.feather \
  --v21-domain --all-neighbours \
  --tags lsst_r lsst_r_extnbr_ho lsst_r_extnbr_indom_tuned lsst_r_extnbr_v21 \
  --output "$NPZ" 2>&1 | grep -v --line-buffered "module command" || { echo RBSUM21_FAILED; exit 1; }

echo; echo "======== SUMMED PER PRIMARY (full 7\" aperture) ========"
python -u scripts/eval_rblend_gap_summed.py --npz "$NPZ" || exit 1
echo; echo "======== SUMMED, restricted to neighbours within 3\" ========"
python -u scripts/eval_rblend_gap_summed.py --npz "$NPZ" --sep-max 3.0 || exit 1
echo RBSUM21_DONE; date
