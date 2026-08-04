#!/bin/bash
#SBATCH --job-name=snc_dec
#SBATCH --time=01:30:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/snc_dec_%j.out
set -o pipefail
# WHERE does the -3.06% small-size label gap enter? 03u showed the two SNC estimators disagree on
# IDENTICAL both-detected objects, so it is not selection -- it is the source catalogues. This splits
# the gap into the sheared-leg and unsheared-leg disagreements and checks the two add back up.
# FIREWALL: half-shear only, no constgold, nothing fit, no `m`.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
echo "### SNC SOURCE DECOMP job=$SLURM_JOB_ID ###"; date
python -u scripts/diag_snc_source_decomp.py \
  --crowd-gs $D/det_meas_crowd_g0.05_val_full.feather \
  --ngmix-gs $D/det_meas_ngmix_g0.05_val.feather \
  --g0-lookup /home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather \
  --ngmix-g0 $D/det_meas_ngmix_g0.0_train.feather \
  --max-case 39 --nominal-g 0.05 --primary-re-min 0.3 --primary-mag-max 26.0 2>&1 | grep -v --line-buffered "module command" || exit 1
echo; echo SNC_DEC_JOB_DONE; date
