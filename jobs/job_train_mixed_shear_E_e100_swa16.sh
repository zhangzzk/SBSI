#!/usr/bin/env bash
# Train a controlled longer arm E: 100 epochs, SWA over the final 16 epochs.

#SBATCH --job-name=sbsi_mixE100
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde

g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$root/det_meas_crowd_conc_g0.05_all200.feather
target=$root/response_target_mixed_train160_6x6x5_matrix.npz
output=$root/measurement_flow_mixed_g0_g005_E_e100_swa16_s501_swabase.pt

export PYTHONPATH=$repo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
unset PYTORCH_CUDA_ALLOC_CONF

for path in "$g0" "$g05" "$target"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
for path in "$output" "${output/swabase/swaavg}" "${output%.pt}_train_curve.npz"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

echo "Training arm=E epochs=100 swa_last_k=16 response_weight=450 components=matrix seed=501"
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
  --max-rows 4000000 --epochs 100 --batch-size 8192 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 \
  --lr 0.0007 --patience 100 --weight-decay 1e-5 --swa-last-k 16 \
  --seed 501 --num-workers 8 --gpu-resident \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --coupling-weight 0 \
  --response-weight 450 --response-delta 0.02 \
  --response-difference central --response-target-npz "$target" \
  --response-components matrix --response-error absolute \
  --response-rel-floor 0.05 --response-bin-ema 0.0
date
echo "MIXED_SHEAR_E100_TRAIN_DONE output=$output"
