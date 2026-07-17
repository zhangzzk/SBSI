#!/bin/bash
#SBATCH --job-name SBSI_SELVAL3
#SBATCH --time=01:30:00
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_selval3.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_selval3.%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"; export OMP_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
CATDIR=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
for GC in "0.05 $CATDIR/det_meas_g0.05_val.feather" "0.2 $CATDIR/det_meas_g0.2_val.feather"; do
  set -- $GC
  echo; echo "### lam150 @ g=$1 ###"
  python -u scripts/validate_selection_response_blend.py --selection-model models/selection_respaware_lam150_v1.pt --catalogue "$2" --nominal-g "$1" --max-rows 8000000
done
