#!/bin/bash
#SBATCH --job-name SBSI_CLOSURE
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_closure_%j.out
#SBATCH --partition=inter
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_closure_%j.err

echo "START - SBSI recovery closure test"
date
eval "$(conda shell.bash hook)"
conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"

MODEL=SBSI/models/measurement_flow_g0_oriented_v1.pt
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.05_val.feather

for S0 in 0.05 0.2; do
  echo "=== CLOSURE s0=$S0 (expect recovered peak ~ $S0) ==="
  python -u SBSI/scripts/validate_heldout_shear_recovery.py \
      --measurement-model "$MODEL" --catalogue "$CAT" \
      --nominal-shear "$S0" --which sheared \
      --closure-shear "$S0" --max-rows 100000 || echo "  (failed)"
done
echo "FINISH"; date
