#!/bin/bash
#SBATCH --job-name=sbsi_tut_exec
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_tutorial_exec_%j.out
set -e
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-master:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export SBSI_CACHE_DIR=models
export BLENDEMU_MODELS=models/blendemu
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI-master
echo "### TUTORIAL_EXEC job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
nvidia-smi -L || true
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -B -u /home/z/Zekang.Zhang/SBSI/jobs/run_tutorial_nb.py examples/sbsi_api_tutorial.ipynb
echo "### TUTORIAL_EXEC_DONE ###"; date
