#!/bin/bash
#SBATCH --job-name=rblend_hc
#SBATCH --time=02:00:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=16
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rblend_hc_%j.out

# Goal-1 iteration-4 (task #32): the WL-core acceptance failures (size_gt1.0 -1.4% z~6,
# mag24-26_sizeGt0.5 ~1%) are R_blend OOS UNDER-FIT of the conditional mean E[r_sim-R_flow|phi]
# in the sparse size/mag TAILS -- NOT an R_flow-grid problem (iteration-3 finer grid was a wash).
# On a phi-defined selection the identity <R_flow+R_blend>=<r_sim> holds iff the GB fit is exact,
# so any residual m IS R_blend fit error. Give the tails their own leaves: higher capacity, lower
# regularization, more training rows. Built on the ACCEPTED iteration-2 _prod R_flow (iteration-3
# R_flow rejected). K-fold-by-case OOS keeps it honest. Experimental; certified untouched.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
RFLOW=$D/rflow_joint_prod_ens3_c40-139.npz                 # ACCEPTED iteration-2 _prod R_flow
RBLEND=$D/rblend_scene_jointrflow_prod_szphiHC_c40-139.npz
echo "### RBLEND_HC job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date

echo "===== high-capacity size-aware R_blend (tail-resolving GB) ====="
stdbuf -oL -eL python -B -u scripts/build_scene_rblend.py \
  --rflow-override $RFLOW --extra-size --nfolds 5 --train-sub 10000000 \
  --gb-iter 800 --gb-leaves 255 --gb-min-leaf 64 --gb-l2 0.3 \
  --out $RBLEND \
  || { echo "RBLEND_HC BUILD FAILED"; exit 1; }

echo "===== re-eval realistic (mag+size) family: _prod R_flow + hi-cap size-aware R_blend ====="
stdbuf -oL -eL python -B -u scripts/eval_selection_robustness.py \
  --rflow-override $RFLOW --rblend-override $RBLEND \
  --realistic --target 0.003 --tag jointprod_szphiHC \
  || { echo "RBLEND_HC EVAL FAILED"; exit 1; }
echo "### RBLEND_HC_DONE job=$SLURM_JOB_ID ###"; date
