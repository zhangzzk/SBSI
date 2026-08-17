#!/bin/bash
#SBATCH --job-name=abl_eval01
#SBATCH --time=00:30:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/abl_eval01_%j.out
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI-ablation
echo "### ABL_EVAL01 job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
echo; echo "########## STEP 0: V1 REPRO (measured-cond, 2D) ##########"
python -B -u scripts/eval_selfresp_gap.py --ckpt-glob '/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s0_v1repro_s*_swaavg.pt'
echo; echo "########## STEP 1: MEASURED->TRUE conditioning ##########"
python -B -u scripts/eval_selfresp_gap.py --ckpt-glob '/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s1_truecond_s*_swaavg.pt'
echo "### ABL_EVAL01_DONE job=$SLURM_JOB_ID ###"; date
