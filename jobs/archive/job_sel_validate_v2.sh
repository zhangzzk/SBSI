#!/bin/bash
#SBATCH --job-name SBSI_SELVALV2
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_selval_v2.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_selval_v2.%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"; export OMP_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI
CATDIR=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
for M in selection_respaware_lam150_v1 selection_respaware_v2_lam150; do
  for GC in "0.05 $CATDIR/det_meas_g0.05_val.feather" "0.2 $CATDIR/det_meas_g0.2_val.feather"; do
    set -- $GC
    echo; echo "############ $M @ g=$1 ############"
    python -u scripts/validate_selection_response_blend.py \
        --selection-model "models/$M.pt" --catalogue "$2" --nominal-g "$1" --target-shear 0.05 --max-rows 8000000
  done
done
