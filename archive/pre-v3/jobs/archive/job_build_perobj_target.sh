#!/bin/bash
#SBATCH --job-name=perobj_tgt
#SBATCH --time=03:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/perobj_tgt_%j.out
set -o pipefail

# BUILD THE PER-OBJECT RESPONSE TARGET that replaces the per-CELL MEAN pin. NO GPU.
#
# WORKLOG 2026-08-03p: five ways of configuring the grid pin (edge allocation, cell resolution,
# mean-head capacity, log-Re input, response weight) are ALL null on the small-size defect, because
# a cell mean is blind to the response ramp inside its own cell.
# WORKLOG 2026-08-03q: removing the binning has a measured ceiling of +0.29 +- 0.82% at small size
# (D6, six shear-free true properties) against the flow's -4.28%. D6 == D7 there, so the extra
# `e_dot_ghat` feature -- which would need a shear direction the g=0 training catalogue does not
# have -- is not required.
#
# The label is the SAME half-shear ruler SNC response, under the SAME domain cuts (mag < 26,
# Re > 0.3) and the same 1/n_pairs weighting, as the grid target it replaces. No new information.
#
# FIREWALL: half-shear ruler only. constgold never read, no `m` consulted.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
SNC=/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather
OUTD=/project/ls-gruen/users/zekang.zhang/sbsi_caches/perobj
mkdir -p "$OUTD"

echo "### BUILD PEROBJ TARGET job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date

python -u scripts/build_perobj_target.py \
  --ruler $D/det_meas_crowd_g0.05_val_full.feather \
  --snc-lookup "$SNC" \
  --nominal-g 0.05 --max-case 99 \
  --primary-mag-max 26.0 --primary-re-min 0.3 \
  ${LABELSRC:+--label-source $LABELSRC} \
  --out ${OUT:-$OUTD/perobj_target_D6_c0-99_dom.joblib} \
  2>&1 | grep -v --line-buffered "module command" || exit 1

echo; echo "PEROBJ_TGT_DONE"; date
