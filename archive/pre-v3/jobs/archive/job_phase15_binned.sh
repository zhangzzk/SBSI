#!/bin/bash
#SBATCH --job-name SBSI_P15_BIN
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_p15_bin.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_p15_bin.%j.err

echo "START - Phase 1.5 binned response resolution (streaming, memory-safe)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

G05=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.05_val.feather

for M in lam300 lam1000; do
  MODEL=models/measurement_flow_g0_shape2d_respbin_${M}_v1.pt
  echo; echo "############################################################"
  echo "##### MODEL: $M"
  echo "############################################################"
  for BINBY in snr mag; do
    echo; echo "=== BINNED RESPONSE (g=0.05, by $BINBY): does R_model track R_sim across bins? ==="
    python -u scripts/binned_response_diagnostic.py \
        --measurement-model "$MODEL" --catalogue "$G05" \
        --nominal-g 0.05 --n-bins 6 --bin-by "$BINBY" \
        --n-samples 64 --max-rows 2000000
  done
done

echo; echo "FINISH"; date
