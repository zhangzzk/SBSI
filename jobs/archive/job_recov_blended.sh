#!/bin/bash
#SBATCH --job-name=AP_recov
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ap_recov_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ap_recov_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
M=models/measurement_flow_g0_ngmix_ap7_respblend_lam1000_v1.pt
for spec in "0.02:/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.02_test.feather" "0.05:/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.05_val.feather"; do
  g="${spec%%:*}"; cat="${spec##*:}"
  echo "===== recovery m @ g=$g (BLENDED subset) ====="
  python -u archive/validate_heldout_shear_recovery.py --measurement-model "$M" \
    --catalogue "$cat" --nominal-shear "$g" --which sheared --blend-subset blended \
    --max-rows 2000000 --output-dir results/ap7_recovery_blended || echo "(failed $g)"
done
echo DONE
