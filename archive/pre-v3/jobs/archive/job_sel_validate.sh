#!/bin/bash
#SBATCH --job-name SBSI_SELVAL
#SBATCH --time=01:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_selval.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_selval.%j.err
echo "START - selection-response validation (old vs new)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /home/z/Zekang.Zhang/SBSI
G05=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.05_val.feather
for M in selection_mlp_g0_shearfree_v1 selection_respaware_lam300_v1; do
  echo; echo "############ $M ############"
  python -u scripts/validate_selection_response_blend.py \
      --selection-model "models/$M.pt" --catalogue "$G05" \
      --nominal-g 0.05 --max-rows 8000000
done
echo; echo "FINISH"; date
