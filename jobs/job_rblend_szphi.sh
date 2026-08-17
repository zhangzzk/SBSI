#!/bin/bash
#SBATCH --job-name=rblend_szphi
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rblend_szphi_%j.out

# Goal-1 size-robustness iteration-2 (task #32): rebuild the scene R_blend for _prod WITH true size
# added to phi (E[r_sim - R_flow | mag, size, neighbours]), OOS K-fold by case, then re-eval the
# realistic (mag+size) family. Diagnosis: R_blend's phi omitted Re_input_p, so it could not model the
# size-dependent SELECTION response -> the +11%/±3% resolved-size residual + the near-PSF cliff.
# Non-circular in the SAME sense the current R_blend already is (OOS-by-case on cert, noted-not-hidden);
# adding a true shear-invariant feature fixes an omission, it does not move constgold into R_flow.
# Experimental/additive; certified m=+0.245% untouched; new tags only.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
RFLOW=$D/rflow_joint_prod_ens3_c40-139.npz
RBLEND=$D/rblend_scene_jointrflow_prod_szphi_c40-139.npz
echo "### RBLEND_SZPHI job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date

echo "===== build size-aware R_blend (phi += Re_input_p), co-calibrated to _prod R_flow ====="
stdbuf -oL -eL python -B -u scripts/build_scene_rblend.py \
  --rflow-override $RFLOW --extra-size --nfolds 5 --out $RBLEND \
  || { echo "RBLEND_SZPHI BUILD FAILED"; exit 1; }

echo "===== re-eval realistic (mag+size) family: _prod R_flow + size-aware R_blend ====="
stdbuf -oL -eL python -B -u scripts/eval_selection_robustness.py \
  --rflow-override $RFLOW --rblend-override $RBLEND \
  --realistic --target 0.003 --tag jointprod_szphi \
  || { echo "RBLEND_SZPHI EVAL FAILED"; exit 1; }
echo "### RBLEND_SZPHI_DONE job=$SLURM_JOB_ID ###"; date
