#!/bin/bash
#SBATCH --job-name SBSI_P15_VAL
#SBATCH --time=03:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_p15_val.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_p15_val.%j.err

echo "START - Phase 1.5 validation (binned R resolution + flow-MLE recovery)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /home/z/Zekang.Zhang/SBSI

G05=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.05_val.feather
G02=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.02_val.feather

for M in lam300 lam1000; do
  MODEL=models/measurement_flow_g0_shape2d_respbin_${M}_v1.pt
  echo; echo "############################################################"
  echo "##### MODEL: $M  ($MODEL)"
  echo "############################################################"

  echo; echo "=== [1] BINNED RESPONSE RESOLUTION (g=0.05, by SNR) -- does R_model track R_sim now? ==="
  python -u scripts/binned_response_diagnostic.py \
      --measurement-model "$MODEL" --catalogue "$G05" \
      --nominal-g 0.05 --n-bins 6 --bin-by snr --n-samples 64 --max-rows 2000000

  echo; echo "=== [1b] BINNED RESPONSE RESOLUTION (g=0.05, by mag) ==="
  python -u scripts/binned_response_diagnostic.py \
      --measurement-model "$MODEL" --catalogue "$G05" \
      --nominal-g 0.05 --n-bins 6 --bin-by mag --n-samples 64 --max-rows 2000000

  echo; echo "=== [2] FLOW-MLE RECOVERY at held-out g=0.02 (sheared half) ==="
  python -u scripts/validate_heldout_shear_recovery.py \
      --measurement-model "$MODEL" --catalogue "$G02" \
      --nominal-shear 0.02 --which sheared \
      --grid-min -0.02 --grid-max 0.08 --grid-n 26 \
      --max-rows 300000 --batch-size 65536
done

echo; echo "FINISH"; date
