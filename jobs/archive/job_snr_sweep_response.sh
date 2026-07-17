#!/bin/bash
#SBATCH --job-name SBSI_SNR_RR
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=96G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_snr_rr_%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_snr_rr_%j.err

# SNR-cut sweep: sim response R_sim(cut), flow first-moment response R_model(cut),
# held-out first-moment m(cut), and pure-forward first-moment m(cut). Tests both
# (1) noise-bias reduction from cutting noisy sources, and (2) selection bias from
# cutting on the shear-dependent measured SNR. Sim-directly + trained-flow in one pass.
echo "START - SNR sweep (response_ratio: sim R_sim + flow first-moment)"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"

MODEL="${MODEL:-SBSI/models/measurement_flow_g0_shape2d_resp_lam1000_v1.pt}"
CATDIR=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
MAXROWS="${MAXROWS:-3000000}"

for SNR in none 10 20 40; do
  echo ""; echo "############################## SNR_MIN = $SNR ##############################"
  ARG=""; [ "$SNR" != "none" ] && ARG="--snr-min $SNR"
  python -u SBSI/scripts/response_ratio_diagnostic.py \
      --measurement-model "$MODEL" \
      --catalogue-005 "$CATDIR/det_meas_g0.05_val.feather" \
      --catalogue-020 "$CATDIR/det_meas_g0.2_val.feather" \
      --max-rows "$MAXROWS" --model-max-rows 400000 $ARG \
      --output-dir SBSI/results/snr_sweep \
      || echo "  (failed SNR=$SNR)"
done
echo "FINISH"; date
