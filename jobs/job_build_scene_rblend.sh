#!/bin/bash
#SBATCH --job-name=scene_rb
#SBATCH --time=01:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/scene_rb_%j.out

# Build a SEPARATE scene-conditioned R_blend = E[r_sim - R_flow | scene phi], OOS K-fold by case.
# DIAGNOSTIC/feasibility: features fixed a priori, trained on the physical response target, never
# on |m| (firewall). Output npz -> eval_selection_robustness.py --rblend-override. R_blend stays
# SEPARATE from the flow (user's constraint).
set -e
source /software/opt/focal/x86_64/python/3.10-2022.08/etc/profile.d/conda.sh
conda activate /project/ls-gruen/users/zekang.zhang/envs/sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd /home/z/Zekang.Zhang/SBSI

echo "### SCENE_RB job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/build_scene_rblend.py "$@" \
  || { echo "SCENE_RB FAILED"; exit 1; }
echo "### SCENE_RB_DONE job=$SLURM_JOB_ID ###"; date
