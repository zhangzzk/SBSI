#!/bin/bash
#SBATCH --job-name=self_tvm
#SBATCH --partition=cip
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/self_tvm_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### SELFRESP_TRUE_VS_MEAS job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/selfresp_true_vs_measured_size.py --max-case 99 2>&1 | grep -vE "module command"
echo DONE; date
