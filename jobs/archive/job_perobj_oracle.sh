#!/bin/bash
#SBATCH --job-name=perobj_or
#SBATCH --time=04:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/perobj_or_%j.out
set -o pipefail

# PRICING THE PER-OBJECT SUPERVISION LEVER (the 2026-07-07 lever, never built). NO GPU.
#
# 15489783 established that the flux x size grid is already fully RESOLVED (36 -> 400 cells moves
# aggregate m by 0.003 pt), so what a per-object target would add is not resolution but
# CONDITIONING: the flow sees 8 true properties, a grid sees the 2-3 it is binned on.
#
# Per-object supervision under a squared-error pull converges to E[R | features], so a regressor on
# the per-object SNC label is what a perfectly per-object-supervised flow would be pulled toward.
#   D2 = trees on (true mag, true Re)   -- the grid's own information; a CONTROL that must land on
#                                          the fine-grid arm C (+0.840%), else D7 is unreadable
#   D7 = trees on all 7 true properties available on both catalogues -- the headroom
# Scored through the SAME eval as arms A/B/B2/C, on the same rows, with the same R_blend.
#
# FIREWALL: label + fit are half-shear ruler only, cases split fit 0-79 / held out 80-99. constgold
# supplies prediction coordinates and the scoring r_sim, nothing else.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
OUTD=/project/ls-gruen/users/zekang.zhang/sbsi_caches/pin_realloc
SNC=/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather
CROWD=/home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather
CONSTCAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
mkdir -p "$OUTD"

echo "### PEROBJ ORACLE job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date

python -u scripts/build_perobj_oracle.py \
  --ruler $D/det_meas_crowd_g0.05_val_full.feather \
  --snc-lookup "$SNC" --constgold "$CONSTCAT" --crowd "$CROWD" \
  --nominal-g 0.05 --max-case 99 --fit-max-case 79 --min-case 40 \
  --primary-mag-max 26.0 --primary-re-min 0.3 \
  --out-prefix $OUTD/oracle_perobj 2>&1 | grep -v --line-buffered "module command" || exit 1

echo; echo "########## SCORED AGAINST EVERY GRID ARM ##########"
python -u scripts/eval_pin_ceiling.py \
  --oracle "A eq-count 6x6=$OUTD/oracle_A_eqcount_6x6.npz" \
  --oracle "C eq-count 20x20=$OUTD/oracle_C_eqcount_20x20.npz" \
  --oracle "D2 trees mag+Re=$OUTD/oracle_perobj_D2.npz" \
  --oracle "D7 trees 7 feats=$OUTD/oracle_perobj_D7.npz" \
  2>&1 | grep -v --line-buffered "module command" || exit 1

echo; echo "PEROBJ_OR_DONE"; date
