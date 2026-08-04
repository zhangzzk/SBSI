#!/bin/bash
#SBATCH --job-name=perobj_cond
#SBATCH --time=04:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/perobj_cond_%j.out
set -o pipefail

# CAN PER-OBJECT SUPERVISION FIX THE SMALL/FAINT DEFECT? NO GPU.
#
# 2026-08-03p exhausted the pin lever (5 attempts, all null; the pin residual saturates). The stated
# mechanism is that a per-CELL MEAN cannot see a ramp INSIDE its own cell. The remaining candidate is
# per-OBJECT supervision, and this prices its CEILING before any retrain.
#
# 03n already priced this lever on aggregate `m` and it looked like a null -- but the flow already
# matches the D7 oracle in aggregate, so the aggregate cannot see the lever at all. The defect is in
# the tails, so this scores the SMALL and FAINT conditionals instead.
#
# Arms differ only in features: D2 (mag,Re = the grid's own info), D3 (+nbr_flux_near, bridging to
# the 3-feature fit of 03k), D7 (all 7). D2 vs flow = what unbinning buys; D7 vs D2 = what extra
# conditioning buys.
#
# Arms are fit on the DUMP's OWN `r_sim_self` and take only FEATURES from the ruler, so fit and score
# are the same quantity by construction. (Job 15501436 fit on the ruler's R_snc instead and its guard
# refused: the two estimators agree in mean to 0.35% but correlate only 0.525, because they subtract
# different g=0 measurements. Legitimate to fit on for a mean residual, but this way is cleaner.)
# Held out by GROUPED 5-FOLD over CASES, so every row gets an out-of-fold prediction and the oracles
# and the flow are scored on identical rows.
#
# FIREWALL: half-shear self-response only; constgold is not read and no `m` is computed.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
SNC=/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather

echo "### PEROBJ CONDITIONAL job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date

python -u scripts/diag_perobj_conditional.py \
  --ruler $D/det_meas_crowd_g0.05_val_full.feather \
  --snc-lookup "$SNC" \
  --dump results/halfshear_selfresp.feather \
  --nominal-g 0.05 --max-case 39 --n-folds 5 \
  --primary-mag-max 26.0 --primary-re-min 0.3 \
  2>&1 | grep -v --line-buffered "module command" || exit 1

echo; echo "PEROBJ_COND_DONE"; date
