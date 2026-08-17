#!/bin/bash
#SBATCH --job-name=triad_hs
#SBATCH --time=01:30:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/triad_hs_%j.out

# HALF-SHEAR Setup-2 (SELECTION) confirmation: is the selection response real, measured the
# sanctioned triad way (intrinsic-e, leg-avg, measured cut) directly off the half-shear g0/g05
# legs? Gate BEFORE pinning R_theta in the flow loss (GOALS.md "validate finite-diff-vs-sim").
# g0 = det_meas_crowd_conc_g0.0 (unsheared); g05 = det_meas_crowd_g0.05 (primary sheared,
# neighbour FIXED so no R_blend leak). Reports R_sel_intr (sanctioned) + R_sel_meas cross-check.
# EVALUATION ONLY on half-shear; constgold is never read here (firewall).
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

NBR=${NBR:-fixed}
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/triad_halfshear_${NBR}.npz
echo "### TRIAD_HS job=$SLURM_JOB_ID node=$SLURMD_NODENAME neighbour=$NBR ###"; date
python -B -u scripts/eval_joint_triad.py \
  --halfshear --hs-neighbour "$NBR" \
  --max-case 99 --true-re-min 0.3 --true-mag-max 26 \
  --hs-size-cuts 2.5 2.9 3.5 4.4 5.5 \
  --hs-mag-cuts 24 24.5 25 \
  --hs-sn-cuts 10 15 20 \
  --n-boot 300 \
  --output "$OUT" \
  || { echo "TRIAD_HS FAILED neighbour=$NBR"; exit 1; }
echo "### TRIAD_HS_DONE job=$SLURM_JOB_ID neighbour=$NBR ###"; date
