#!/bin/bash
#SBATCH --job-name=build_hsflow
#SBATCH --time=01:30:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/build_hsflow_%j.out

# Rebuild the joint-flow SHAPE (FLOW) training catalogue INCLUDING isolated objects, from the
# two ngmix half-sim legs (g=0 -> _m, g=0.05 -> _p).  Fixes the 0%-isolated gap that inflates
# eval B isolated cells (cont.57/58).  Convention identical to certified r_sim (measured_e1_m ==
# measured_ngmix_g1, verified).  ADDITIVE/experimental; never wired into certified m.
set -e
source /software/opt/focal/x86_64/python/3.10-2022.08/etc/profile.d/conda.sh
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/z/Zekang.Zhang/SBSI

OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/self_response_halfsim_isoblend_cases0_99.feather
echo "### BUILD_HSFLOW job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/build_halfsim_flow_catalogue.py --max-case 100 --out "$OUT" \
  || { echo "BUILD_HSFLOW FAILED"; exit 1; }
echo "### BUILD_HSFLOW_DONE job=$SLURM_JOB_ID ###"; date
