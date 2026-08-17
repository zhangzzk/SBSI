#!/bin/bash
#SBATCH --job-name SBSI_BLVAL
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_blval.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_blval.%j.err
echo "START - blend-fix validation (probes on respblend models)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /home/z/Zekang.Zhang/SBSI
G05=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.05_val.feather
for M in lam300 lam1000; do
  MODEL=models/measurement_flow_g0_shape2d_respblend_${M}_v1.pt
  echo; echo "##################### MODEL: respblend_$M #####################"
  echo "--- [1] isolated vs blended (should now both be ~1.0) ---"
  python -u scripts/blend_response_diagnostic.py --measurement-model "$MODEL" \
      --catalogue "$G05" --nominal-g 0.05 --n-samples 64 --max-rows 2500000
  echo "--- [3] resolution across blend severity (R_model should now RISE with distance, like R_sim) ---"
  python -u scripts/blend_resolution.py --measurement-model "$MODEL" \
      --catalogue "$G05" --nominal-g 0.05 --bin-by distance --n-bins 5 --n-samples 64 --max-rows 2500000
  echo "--- [brightness check] SNR-binned (must STILL track R_sim; don't regress) ---"
  python -u scripts/binned_response_diagnostic.py --measurement-model "$MODEL" \
      --catalogue "$G05" --nominal-g 0.05 --n-bins 6 --bin-by snr --n-samples 64 --max-rows 2000000
done
echo; echo "FINISH"; date
