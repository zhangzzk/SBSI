#!/bin/bash
#SBATCH --job-name=rflow_orac
#SBATCH --time=00:40:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rflow_orac_%j.out

# cont.90 (ceiling test, NO GPU): harvest the FINE non-circular self-response target
# (isoblend_snc_6x6x5) per-object as an R_flow override = "perfect binding to the fine target".
# Eval it through the real acceptance harness => the EXACT ceiling of the higher-lambda retrain
# lever (a trained flow can only approach, never beat, its response target). If this is not
# sub-percent, closing the binding gap via lambda CANNOT reach sub-percent on the current target.
# NO --isolated-zero (this is R_flow, nonzero for isolated). constgold = coords only, r_sim never read.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
CONSTCAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
GRID=results/response_target_isoblend_snc_c0-99_6x6x5.npz
OVR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/rflow_oracle_isoblend_c40-139.npz
echo "### RFLOW_ORAC job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -B -u scripts/harvest_grid_perobj.py \
  --grid $GRID --catalogue $CONSTCAT --min-case 40 \
  --out $OVR 2>&1 | grep -v "module command"
echo "RFLOW_ORAC_DONE ovr=$OVR"; date
