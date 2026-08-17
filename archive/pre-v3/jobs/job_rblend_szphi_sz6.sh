#!/bin/bash
#SBATCH --job-name=rblend_sz6
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rblend_sz6_%j.out

# Goal-1 iteration-3 tail (task #32): size-aware R_blend co-calibrated to the _prodsz6 (finer-size)
# R_flow, then re-eval the realistic (mag+size) family. Head-to-head vs jointprod_szphi to see if
# finer R_flow size supervision shrinks the residual near-PSF bias. Experimental; certified untouched.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
RFLOW=$D/rflow_joint_prodsz6_ens3_c40-139.npz
RBLEND=$D/rblend_scene_jointrflow_prodsz6_szphi_c40-139.npz
echo "### RBLEND_SZ6 job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date

echo "===== size-aware R_blend (phi += Re_input_p) co-calibrated to _prodsz6 R_flow ====="
stdbuf -oL -eL python -B -u scripts/build_scene_rblend.py \
  --rflow-override $RFLOW --extra-size --nfolds 5 --out $RBLEND \
  || { echo "RBLEND_SZ6 BUILD FAILED"; exit 1; }

echo "===== re-eval realistic (mag+size) family: _prodsz6 R_flow + size-aware R_blend ====="
stdbuf -oL -eL python -B -u scripts/eval_selection_robustness.py \
  --rflow-override $RFLOW --rblend-override $RBLEND \
  --realistic --target 0.003 --tag prodsz6_szphi \
  || { echo "RBLEND_SZ6 EVAL FAILED"; exit 1; }
echo "### RBLEND_SZ6_DONE job=$SLURM_JOB_ID ###"; date
