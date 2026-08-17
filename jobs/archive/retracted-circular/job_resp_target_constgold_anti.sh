#!/bin/bash
#SBATCH --job-name=rtgt_cg
#SBATCH --time=01:30:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=10
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/rtgt_cg_%j.out

# METRIC-CONSISTENT response target: build R_sim(flux x size x blend) from the CONSTGOLD antithetic
# +/-g renders (the SAME sample + central-diff scheme as the acceptance metric's r_sim), so the flow's
# response target matches the validation truth (fixes the +6..8% large-size selection bias diagnosed
# in cont.63 as a train/eval response-definition inconsistency: variable-shear forward-diff grid ~0.916
# vs constgold central-diff metric ~1.02 at large iso size).  6x6 flux x size, isolated + 4 distance
# bins (5 blend bins) -- SAME structure train_forward_prototype expects.  Cases 0-99 (flow train range).
# FIREWALL: a-priori physics choice (use the metric's own response definition), OOS by case, not |m|-tuning.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
CG=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
OUT=results/response_target_constgold_c0-99_6x6x5.npz
echo "### RTGT_CG job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/compute_response_target_blend.py \
  --catalogue "$CG" --antithetic --nominal-g 0.02 \
  --n-flux 6 --n-size 6 --n-dist 4 \
  --max-case 99 --min-count 200 \
  --output "$OUT" \
  || { echo "RTGT_CG FAILED"; exit 1; }
echo "### RTGT_CG_DONE job=$SLURM_JOB_ID ###"; ls -la "$OUT"; date
