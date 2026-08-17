#!/bin/bash
#SBATCH --job-name=rbverdict
#SBATCH --time=03:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=12
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rbverdict_%j.out

# FLOW vs EMULATOR on the faint shell, settled with the PER-PAIR ruler. Independent re-run.
#
# Three steps, one job:
#   (1) eval_rblend_gap_measured.py  -- per-PAIR R_blend truth vs emulator, binned by the PRIMARY'S
#       MEASURED magnitude / S/N / size, 45-deg null PER BIN. ap7 legs, every annotated neighbour
#       kept, cluster-robust errors with the primary as the cluster.
#   (2) eval_rblend_gap_summed.py    -- per-PAIR -> per-PRIMARY SUMMED R_blend, which is the quantity
#       `m` actually uses (build_blend_lookup sums the emulator over a primary's neighbours).
#   (3) eval_rblend_ruler_coverage.py -- NEW. What FRACTION of the deployed R_blend (k=20, r_max=10")
#       the ruler's 7" annotation actually certifies. Turns the headline "cannot settle" caveat into
#       a number. All radii <= the trained 10", so no extrapolation.
#
# FIREWALL: half-shear legs and half-shear input fields only. constgold is never opened. Nothing is
# trained, fitted, selected or tuned.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
C=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/rblend_verdict; mkdir -p $D
NPZ=$D/rblend_gap_measured_ap7_v2.npz
TAGS="lsst_r_extnbr_indom_tuned lsst_r_extnbr_ho"

echo "### RBLEND VERDICT  job=$SLURM_JOB_ID ###"; date

echo; echo "##### STEP 1: per-pair ruler binned by MEASURED props (ap7, all neighbours) #####"
python -u scripts/eval_rblend_gap_measured.py \
  --gs-leg $C/det_meas_ngmix_ap7_g0.05_val.feather \
  --g0-leg $C/det_meas_ngmix_ap7_g0.0_train.feather \
  --all-neighbours --true-re-min 0.3 --true-mag-max 26.0 \
  --tags $TAGS --output "$NPZ" || exit 1

echo; echo "##### STEP 2: SUMMED per-primary R_blend (7\" = the full ap7 annotation) #####"
python -u scripts/eval_rblend_gap_summed.py --npz "$NPZ" || exit 1

echo; echo "##### STEP 2b: SUMMED, restricted to neighbours within 3\" (nn3-comparable) #####"
python -u scripts/eval_rblend_gap_summed.py --npz "$NPZ" --sep-max 3.0 || exit 1

echo; echo "RBVERDICT_DONE"; date
# STEP 3 (aperture coverage) runs as its own job, jobs/job_rblend_coverage.sh -- it is model-only and
# does not depend on this dump, so it is submitted in parallel rather than serialised behind it.
