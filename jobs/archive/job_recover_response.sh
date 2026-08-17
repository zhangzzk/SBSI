#!/bin/bash
#SBATCH --job-name SBSI_REC_RESP
#SBATCH --time=03:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_rec_resp_%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_rec_resp_%j.err

echo "START - response-aware flow recovery (flow-MLE m) on g=0.05/0.2/0.02"
date
eval "$(conda shell.bash hook)"
conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"

MODEL="${MODEL:?set MODEL=SBSI/models/...pt}"
CATDIR=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
MAXROWS="${MAXROWS:-1000000}"

run() {  # cat nominal
  echo "########## recovery shear=$2  model=$(basename $MODEL) ##########"
  python -u SBSI/scripts/validate_heldout_shear_recovery.py \
      --measurement-model "$MODEL" \
      --catalogue "$1" --nominal-shear "$2" --which sheared \
      --max-rows "$MAXROWS" \
      || echo "  (run failed)"
}

run "$CATDIR/det_meas_g0.02_val.feather" 0.02
run "$CATDIR/det_meas_g0.05_val.feather" 0.05
run "$CATDIR/det_meas_g0.2_val.feather"  0.2

echo "FINISH"; date
