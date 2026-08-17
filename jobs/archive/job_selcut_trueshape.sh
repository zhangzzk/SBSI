#!/bin/bash
#SBATCH --job-name SBSI_SELTRUE
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_seltrue.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_seltrue.%j.err

echo "START - selection bias of measured-flux cut (true-shape, single catalogue)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

echo "##### g=0.05 #####"
python -u scripts/selection_cut_trueshape.py \
    --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.05_val.feather \
    --nominal-g 0.05 --max-rows 20000000
echo "##### g=0.02 (cross-check, larger 1/g amplification) #####"
python -u scripts/selection_cut_trueshape.py \
    --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.02_val.feather \
    --nominal-g 0.02 --max-rows 20000000

echo; echo "FINISH"; date
