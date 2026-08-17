#!/bin/bash
#SBATCH --job-name SBSI_SELTGT
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_seltgt.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_seltgt.%j.err
echo "START - blend-aware selection-response target"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
python -u scripts/compute_selection_target_blend.py \
    --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.05_val.feather \
    --nominal-g 0.05 --n-flux 4 --n-size 2 --n-dist 3 --max-rows 20000000 \
    --output results/selection_target_g0.05_4x2x4_blend.npz
echo; echo "FINISH"; date
