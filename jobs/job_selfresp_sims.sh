#!/bin/bash
#SBATCH --job-name=selfresp
#SBATCH --time=01:00:00
#SBATCH --mem=220G
#SBATCH --cpus-per-task=12
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/selfresp_%j.out

# Direct sim-vs-sim confirmation (owner): half-shear vs constgold ISOLATED faint self-response,
# identical cuts, with bootstrap uncertainty + distribution check. CPU/IO only.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI
echo "### SELFRESP job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -B -u scripts/compare_selfresp_sims.py --hs-max-case 19 --cg-max-case 39 \
  || { echo "SELFRESP FAILED"; exit 1; }
echo "### SELFRESP_DONE job=$SLURM_JOB_ID ###"; date
