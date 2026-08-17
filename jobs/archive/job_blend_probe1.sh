#!/bin/bash
#SBATCH --job-name SBSI_BLEND1
#SBATCH --time=01:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_blend1.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_blend1.%j.err

echo "START - blend probe 1: isolated vs blended response (sim vs flow)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

G05=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.05_val.feather
for M in lam300 lam1000; do
  echo; echo "########## MODEL: $M ##########"
  python -u scripts/blend_response_diagnostic.py \
      --measurement-model models/measurement_flow_g0_shape2d_respbin_${M}_v1.pt \
      --catalogue "$G05" --nominal-g 0.05 --n-samples 64 --max-rows 2500000
done
echo; echo "FINISH"; date
