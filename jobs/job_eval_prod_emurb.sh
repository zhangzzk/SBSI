#!/bin/bash
#SBATCH --job-name=prod_emurb
#SBATCH --time=00:40:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/prod_emurb_%j.out

# cont.81 clean non-circular product (task #29/#32): the never-evaluated headline combo =
#   _prod joint R_flow (det_meas legs, non-circular)  +  LEGACY per-pair emulator R_blend
# (the dump's DEFAULT R_blend column = blend_lookup_extnbrho, constgold-FREE). NO --rblend-override
# so it uses that emulator R_blend. constgold r_sim is the VALIDATION truth only. This is the honest
# number for the improved R_flow under the owner's R_blend firewall (project_rblend_firewall).
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
RFLOW=$D/rflow_joint_prod_ens3_c40-139.npz
echo "### PROD_EMURB job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
echo "R_flow=_prod joint ensemble; R_blend=DUMP DEFAULT (legacy emulator, constgold-free); NO rblend override"

stdbuf -oL -eL python -B -u scripts/eval_selection_robustness.py \
  --rflow-override $RFLOW \
  --realistic --target 0.003 --tag jointprod_emurb \
  || { echo "PROD_EMURB EVAL FAILED"; exit 1; }
echo "### PROD_EMURB_DONE job=$SLURM_JOB_ID ###"; date
