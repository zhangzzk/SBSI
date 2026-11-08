#!/usr/bin/env bash
# Controlled Flow-E retrains: historical 450/500 and matched 500/500 pilot.

#SBATCH --job-name=sbsi_Ephot
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric

g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$source_root/det_meas_crowd_conc_g0.05_all200.feather
response=$source_root/response_target_mixed_train160_6x6x5_matrix.npz
coupling=$root/coupling_target_mixed_train160_6x6x5.npz

case "${SLURM_ARRAY_TASK_ID:?submit as array 0-1}" in
  0) response_weight=450; tag=E_r450_t500 ;;
  1) response_weight=500; tag=E_r500_t500 ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
output=$root/measurement_flow_mixed_g0_g005_${tag}_s501_swabase.pt

export PYTHONPATH=$repo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
unset PYTORCH_CUDA_ALLOC_CONF

for path in "$g0" "$g05" "$response" "$coupling"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
for path in "$output" "${output/swabase/swaavg}" "${output%.pt}_train_curve.npz"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

echo "Training $tag response_weight=$response_weight coupling_weight=500 seed=501"
date
nvidia-smi -L
"$python" -u -m sbsi.flow_training \
  --catalogue "$g0" --additional-catalogue "$g05" --output "$output" \
  --target-column detected --selection-name sextractor_detected \
  --feature-set g0_meas_crowd_conc_szfl_noz \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type mean_affine --mean-hidden 128 \
  --flow-blind-features e1_input_p e2_input_p \
  --fold-catalogue-shear --max-cases 200 --expected-case-count 200 \
  --validation-group-column case --validation-size 0.2 \
  --max-rows 4000000 --epochs 80 --batch-size 8192 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 \
  --lr 0.0007 --patience 10 --weight-decay 1e-5 --swa-last-k 8 \
  --seed 501 --num-workers 8 --gpu-resident \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --coupling-weight 500 --coupling-target-npz "$coupling" \
  --response-weight "$response_weight" --response-delta 0.02 \
  --response-difference central --response-target-npz "$response" \
  --response-components matrix --response-error absolute \
  --response-rel-floor 0.05 --response-bin-ema 0.0
date
echo "MIXED_SHEAR_E_PHOTOMETRIC_TRAIN_DONE tag=$tag output=$output"
