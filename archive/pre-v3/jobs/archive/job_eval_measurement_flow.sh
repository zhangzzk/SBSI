#!/bin/bash
#SBATCH --job-name SBSI_EVALFLOW
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_evalflow_%j.out
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_evalflow_%j.err
#SBATCH --partition=inter

echo "START - SBSI measurement-flow eval (M diagnostic + recovery)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"

MODEL="${MODEL:-SBSI/models/measurement_flow_g0_oriented_v2.pt}"
CATDIR=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues

echo "### shape-response diagnostic ###"
python -u SBSI/scripts/diagnose_shape_response.py --measurement-model "$MODEL" || echo "(diag failed)"

for cfg in "0.05 $CATDIR/det_meas_g0.05_val.feather" "0.2 $CATDIR/det_meas_g0.2_val.feather"; do
  set -- $cfg
  echo "### recovery shear=$1 ###"
  python -u SBSI/scripts/validate_heldout_shear_recovery.py \
      --measurement-model "$MODEL" --catalogue "$2" \
      --nominal-shear "$1" --which sheared --max-rows 100000 || echo "(recovery failed)"
done
echo "FINISH"; date
