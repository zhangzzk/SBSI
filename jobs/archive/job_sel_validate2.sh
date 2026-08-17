#!/bin/bash
#SBATCH --job-name SBSI_SELVAL2
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_selval2.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_selval2.%j.err
echo "START - selection validation: lam300 vs lam1000, g=0.05 and g=0.2 (transfer)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
CATDIR=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
for M in selection_respaware_lam300_v1 selection_respaware_lam1000_v1; do
  for GC in "0.05 $CATDIR/det_meas_g0.05_val.feather" "0.2 $CATDIR/det_meas_g0.2_val.feather"; do
    set -- $GC; G=$1; CAT=$2
    echo; echo "############ $M  @ g=$G ############"
    python -u scripts/validate_selection_response_blend.py \
        --selection-model "models/$M.pt" --catalogue "$CAT" --nominal-g "$G" --max-rows 8000000
  done
done
echo; echo "FINISH"; date
