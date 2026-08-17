#!/bin/bash
#SBATCH --job-name SBSI_RESPRATIO
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_respratio_%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_respratio_%j.err

echo "START - SBSI response-ratio gating diagnostic (model vs sim first-moment response)"
date
eval "$(conda shell.bash hook)"
conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"

CATDIR=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
C005="$CATDIR/det_meas_g0.05_val.feather"
C020="$CATDIR/det_meas_g0.2_val.feather"
MAXROWS="${MAXROWS:-4000000}"
MODELMAXROWS="${MODELMAXROWS:-400000}"

for M in ${MODELS:-measurement_flow_g0_shape2d_meanblind_v1 measurement_flow_g0_shape2d_meanmlp_v1}; do
  echo "########################## MODEL $M ##########################"
  python -u SBSI/scripts/response_ratio_diagnostic.py \
      --measurement-model "SBSI/models/$M.pt" \
      --catalogue-005 "$C005" --catalogue-020 "$C020" \
      --max-rows "$MAXROWS" --model-max-rows "$MODELMAXROWS" --n-samples 64 \
      --output-dir SBSI/results/response_ratio \
      || echo "  (failed: $M)"
done

echo "FINISH"; date
