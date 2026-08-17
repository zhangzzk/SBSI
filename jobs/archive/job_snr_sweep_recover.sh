#!/bin/bash
#SBATCH --job-name SBSI_SNR_REC
#SBATCH --time=05:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_snr_rec_%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_snr_rec_%j.err

# SNR-cut sweep for the FLOW-MLE estimator (marginal-likelihood recovery).
echo "START - SNR sweep (flow-MLE recovery)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"

MODEL="${MODEL:-SBSI/models/measurement_flow_g0_shape2d_resp_lam1000_v1.pt}"
CATDIR=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
MAXROWS="${MAXROWS:-600000}"

rec() {  # cat g snr
  local arg=""; [ "$3" != "none" ] && arg="--snr-min $3"
  echo "########## recovery g=$2  SNR_MIN=$3 ##########"
  python -u SBSI/scripts/validate_heldout_shear_recovery.py \
      --measurement-model "$MODEL" --catalogue "$1" --nominal-shear "$2" --which sheared \
      --max-rows "$MAXROWS" $arg || echo "  (failed g=$2 snr=$3)"
}

for SNR in none 10 20 40; do
  rec "$CATDIR/det_meas_g0.05_val.feather" 0.05 "$SNR"
  rec "$CATDIR/det_meas_g0.2_val.feather"  0.2  "$SNR"
done
echo "FINISH"; date
