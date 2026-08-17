#!/bin/bash
#SBATCH --job-name=cg_mdlbsz
#SBATCH --time=00:30:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_mdlbsz_%j.out

# MODEL (detection classifier) predicted response vs beta per size bin, on det_meas parent,
# reusing the sim run's (size x beta) edges. For the sim-vs-model overlay.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI-ablation
echo "### CG MDLBSZ job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_model_beta_bysize.py \
  --edges-npz /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/detection_beta_bysize_v2.npz \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/model_beta_bysize_v2.npz
echo "CG_MDLBSZ_JOB_DONE"; date
