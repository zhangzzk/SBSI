#!/bin/bash
#SBATCH --job-name SBSI_P15_SLOPE
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_p15_slope.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_p15_slope.%j.err

echo "START - Phase 1.5 flow-MLE 2-shear slope (disentangle m vs c)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

G02=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.02_val.feather
G05=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.05_val.feather

for M in lam300 lam1000; do
  MODEL=models/measurement_flow_g0_shape2d_respbin_${M}_v1.pt
  echo; echo "########## MODEL: $M ##########"

  echo; echo "=== flow-MLE recovery @ g=0.02 ==="
  python -u scripts/validate_heldout_shear_recovery.py \
      --measurement-model "$MODEL" --catalogue "$G02" \
      --nominal-shear 0.02 --which sheared \
      --grid-min -0.02 --grid-max 0.08 --grid-n 26 \
      --max-rows 300000 --batch-size 65536

  echo; echo "=== flow-MLE recovery @ g=0.05 ==="
  python -u scripts/validate_heldout_shear_recovery.py \
      --measurement-model "$MODEL" --catalogue "$G05" \
      --nominal-shear 0.05 --which sheared \
      --grid-min -0.01 --grid-max 0.11 --grid-n 31 \
      --max-rows 300000 --batch-size 65536
done

echo; echo "FINISH"; date
