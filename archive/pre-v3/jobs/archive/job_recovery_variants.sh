#!/bin/bash
#SBATCH --job-name SBSI_REC_VAR
#SBATCH --time=03:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_rec_var_%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_rec_var_%j.err

echo "START - SBSI recovery variants (baseline / SNR>20 / isolated / both)"
date
eval "$(conda shell.bash hook)"
conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"

MODEL="${MODEL:-SBSI/models/measurement_flow_g0_shape2d_meanblind_v1.pt}"
CATDIR=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
MAXROWS="${MAXROWS:-1000000}"
# Model-specific output dir so concurrent jobs for different models cannot collide.
MODELTAG=$(basename "$MODEL" .pt)
OUTDIR="SBSI/results/heldout_shear_recovery/$MODELTAG"
mkdir -p "$OUTDIR"
echo "MODEL=$MODEL  OUTDIR=$OUTDIR  MAXROWS=$MAXROWS"

run() {  # nominal catalogue extra_args...
  local nominal="$1"; local cat="$2"; shift 2
  python -u SBSI/scripts/validate_heldout_shear_recovery.py \
      --measurement-model "$MODEL" --catalogue "$cat" \
      --nominal-shear "$nominal" --which sheared \
      --output-dir "$OUTDIR" \
      --max-rows "$MAXROWS" "$@" || echo "  (failed: $nominal $*)"
}

for COMBO in ${COMBOS:-baseline snr20 isolated snr20_isolated}; do
  case "$COMBO" in
    baseline)       ARGS="" ;;
    snr20)          ARGS="--snr-min 20" ;;
    isolated)       ARGS="--blend-subset isolated" ;;
    snr20_isolated) ARGS="--snr-min 20 --blend-subset isolated" ;;
  esac
  echo "########## combo=$COMBO  args=[$ARGS] ##########"
  run 0.05 "$CATDIR/det_meas_g0.05_val.feather" $ARGS
  run 0.2  "$CATDIR/det_meas_g0.2_val.feather"  $ARGS
done

echo "FINISH"; date
