#!/bin/bash
#SBATCH --job-name=sbsi_tutorial
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_tutorial_%j.out
set -e
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-master:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI-master
echo "### TUTORIAL_NB job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
nvidia-smi -L || true
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -B -u /home/z/Zekang.Zhang/SBSI/jobs/run_tutorial_nb.py examples/sbsi_api_tutorial.ipynb
echo "### TUTORIAL_NB_DONE ###"; date
