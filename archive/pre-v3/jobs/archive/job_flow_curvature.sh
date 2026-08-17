#!/bin/bash
#SBATCH --job-name SBSI_CURV
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_curv.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_curv.%j.err

echo "START - flow response curvature vs sim curvature (by true-mag cut)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /home/z/Zekang.Zhang/SBSI

G05=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.05_val.feather
G02=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.02_val.feather

for M in lam300 lam1000; do
  echo; echo "########## MODEL: $M ##########"
  python -u scripts/flow_response_curvature.py \
      --measurement-model models/measurement_flow_g0_shape2d_respbin_${M}_v1.pt \
      --cat-005 "$G05" --cat-002 "$G02" --max-rows 3000000
done

echo; echo "FINISH"; date
