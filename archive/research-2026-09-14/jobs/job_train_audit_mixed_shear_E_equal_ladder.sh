#!/usr/bin/env bash
# Equal response/photometric-loss ladder for Flow E, including a pure-NLL control,
# followed by a common held-out audit.

#SBATCH --job-name=sbsi_Eeq
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-6%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_equal_ladder_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_equal_ladder_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
train_python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
audit_python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
target_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_equal_ladder_v1

case "${SLURM_ARRAY_TASK_ID:?submit as array 0-6}" in
  0) weight=5 ;;
  1) weight=10 ;;
  2) weight=25 ;;
  3) weight=50 ;;
  4) weight=75 ;;
  5) weight=100 ;;
  6) weight=0 ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

tag="r${weight}_t${weight}"
g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$source_root/det_meas_crowd_conc_g0.05_all200.feather
response=$source_root/response_target_mixed_train160_6x6x5_matrix.npz
coupling_train=$target_root/coupling_target_mixed_train160_6x6x5.npz
coupling_validation=$target_root/coupling_target_mixed_validation40_6x6x5.npz
output=$root/measurement_flow_mixed_g0_g005_E_${tag}_s501_swabase.pt
swa=${output/swabase/swaavg}
audit=$root/heldout_derivative_losses_E_${tag}_s501.json

export PYTHONPATH=$repo
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "$root/logs"
for path in "$g0" "$g05" "$response" "$coupling_train" "$coupling_validation"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
for path in "$output" "$swa" "${output%.pt}_train_curve.npz" "$audit"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

echo "Training Flow-E response_weight=$weight coupling_weight=$weight seed=501"
date
nvidia-smi -L
"$train_python" -u -m sbsi.flow_training \
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
  --coupling-weight "$weight" --coupling-target-npz "$coupling_train" \
  --response-weight "$weight" --response-delta 0.02 \
  --response-difference central --response-target-npz "$response" \
  --response-components matrix --response-error absolute \
  --response-rel-floor 0.05 --response-bin-ema 0.0

echo "Auditing final SWA checkpoint $swa"
"$audit_python" -u "$repo/scripts/audit_flow_derivative_losses.py" \
  --model "$swa" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --coupling-target "$coupling_validation" \
  --output "$audit" --device cuda --delta 0.02

date
echo "FLOW_E_EQUAL_LADDER_DONE tag=$tag model=$swa audit=$audit"
