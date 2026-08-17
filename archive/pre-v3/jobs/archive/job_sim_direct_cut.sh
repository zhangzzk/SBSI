#!/bin/bash
#SBATCH --job-name SBSI_CUTBIAS
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_cutbias.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_cutbias.%j.err

echo "START - model-free sim-direct cut-bias (measured-SNR vs true-mag cuts)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

python -u scripts/sim_direct_cut_bias.py \
    --cat-005 /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.05_val.feather \
    --cat-002 /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.02_val.feather \
    --max-rows 12000000

echo; echo "FINISH"; date
