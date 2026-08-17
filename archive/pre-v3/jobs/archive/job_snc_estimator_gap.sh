#!/bin/bash
#SBATCH --job-name=snc_gap
#SBATCH --time=01:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/snc_gap_%j.out
set -o pipefail
# Do the two SNC self-response estimators agree REGION BY REGION? The pin is fit to the ruler's
# R_snc; every score is graded on the dump's r_sim_self. 03q compared them in AGGREGATE only.
# See the script docstring for why (1) model==target, (2) model!=truth, (3) oracle==truth can only
# all hold if these two disagree at small size. FIREWALL: half-shear only, no constgold, no `m`.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
SNC=/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather
echo "### SNC ESTIMATOR GAP job=$SLURM_JOB_ID ###"; date
python -u scripts/diag_snc_estimator_gap.py \
  --ruler $D/det_meas_crowd_g0.05_val_full.feather --snc-lookup "$SNC" \
  --dump results/halfshear_selfresp.feather \
  --nominal-g 0.05 --max-case 39 --primary-mag-max 26.0 --primary-re-min 0.3 \
  2>&1 | grep -v --line-buffered "module command" || exit 1
echo; echo SNC_GAP_JOB_DONE; date
