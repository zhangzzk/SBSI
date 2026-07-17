#!/bin/bash
#SBATCH --job-name SBSI_MDIAG
#SBATCH --time=01:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_mdiag_%j.out
#SBATCH --partition=inter
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_mdiag_%j.err

echo "START - SBSI M_model vs M_data diagnostic"
date

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.0_train.feather

for MODEL in \
  SBSI/models/measurement_flow_g0_shape2d_olspin_v1.pt \
  SBSI/models/measurement_flow_g0_shape2d_meanblind_v1.pt \
  SBSI/models/measurement_flow_g0_shape2d_mean_v1.pt \
  SBSI/models/measurement_flow_g0_shape2d_v1.pt ; do
  echo ""
  echo "########## M_model for: $MODEL ##########"
  python -u SBSI/scripts/diagnose_shape_response.py \
      --measurement-model "$MODEL" \
      --catalogue "$CAT" \
      --max-rows 300000 \
      --flow-rows 8000 \
      --n-samples 48 \
    || echo "  (diagnostic failed for $MODEL)"
done

echo "FINISH"
date
