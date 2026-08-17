#!/bin/bash
#SBATCH --job-name SBSI_RECOVER
#SBATCH --time=03:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_recover_%j.out
#SBATCH --partition=inter
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_recover_%j.err

echo "START - SBSI held-out-shear recovery"
date

eval "$(conda shell.bash hook)"
conda activate sims1

export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"

MODEL="${MODEL:-SBSI/models/measurement_flow_g0_oriented_v1.pt}"
CATDIR=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
MAXROWS="${MAXROWS:-100000}"
SNRMIN="${SNRMIN:-}"
MAGMAX="${MAGMAX:-}"

CUTARGS=""
[ -n "$SNRMIN" ] && CUTARGS="$CUTARGS --snr-min $SNRMIN"
[ -n "$MAGMAX" ] && CUTARGS="$CUTARGS --mag-max $MAGMAX"

run() {  # cat nominal which
  echo "=== recovery: shear=$2 which=$3  cut:[$CUTARGS] ==="
  python -u SBSI/scripts/validate_heldout_shear_recovery.py \
      --measurement-model "$MODEL" \
      --catalogue "$1" \
      --nominal-shear "$2" \
      --which "$3" \
      --max-rows "$MAXROWS" \
      $CUTARGS \
      || echo "  (run failed; see error above)"
}

# Sheared recovery (g=0.05 and g=0.2) with whatever quality cut is set.
run "$CATDIR/det_meas_g0.05_val.feather" 0.05 sheared
run "$CATDIR/det_meas_g0.2_val.feather"  0.2  sheared

echo "FINISH"
date
