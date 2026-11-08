#!/usr/bin/env bash
# Directly regularize derivatives of a plain flow's sampled conditional mean.
# Two weights bracket a mild and stronger response constraint; both warm-start
# from the identical no-mean-head pure-NLL checkpoint.

#SBATCH --job-name=sbsi_Edirect
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
train_python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
audit_python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
photo_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric
parent_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_plain_full_condition_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_v1

case "${SLURM_ARRAY_TASK_ID:?submit as array 0-1}" in
  0) weight=1 ;;
  1) weight=5 ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$source_root/det_meas_crowd_conc_g0.05_all200.feather
response=$source_root/response_target_mixed_train160_6x6x5_matrix.npz
coupling_train=$photo_root/coupling_target_mixed_train160_6x6x5.npz
coupling_validation=$photo_root/coupling_target_mixed_validation40_6x6x5.npz
parent=$parent_root/measurement_flow_mixed_g0_g005_E_plain_affine_pure_nll_fullcond_s501_swaavg.pt
output=$root/measurement_flow_plain_direct_r${weight}_t${weight}_d4_s501_swabase.pt
swa=${output/swabase/swaavg}
audit=$root/complete_response_plain_direct_r${weight}_t${weight}_d4_s501.json

mkdir -p "$root/logs"
for path in "$g0" "$g05" "$response" "$coupling_train" "$coupling_validation" "$parent"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
for path in "$output" "$swa" "${output%.pt}_train_curve.npz" "$audit"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

export PYTHONPATH=$repo
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
unset PYTORCH_CUDA_ALLOC_CONF

date
nvidia-smi -L
echo "Fine-tuning plain flow with direct complete-flow derivatives: weight=$weight draws=4"
"$train_python" -u -m sbsi.flow_training \
  --catalogue "$g0" --additional-catalogue "$g05" --output "$output" \
  --target-column detected --selection-name sextractor_detected \
  --feature-set g0_meas_crowd_conc_szfl_noz \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type affine \
  --fold-catalogue-shear --max-cases 200 --expected-case-count 200 \
  --validation-group-column case --validation-size 0.2 \
  --max-rows 4000000 --epochs 20 --batch-size 4096 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 \
  --lr 0.0001 --patience 6 --weight-decay 0 --swa-last-k 5 \
  --seed 501 --num-workers 8 --gpu-resident \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --fine-tune-reference "$parent" --nll-weight 1 --preserve-weight 0 \
  --response-estimator complete_flow --response-draws 4 --response-sample-seed 7301 \
  --response-weight "$weight" --response-delta 0.02 \
  --response-difference central --response-target-npz "$response" \
  --response-components matrix --response-error absolute \
  --response-rel-floor 0.05 --response-bin-ema 0 \
  --coupling-weight "$weight" --coupling-target-npz "$coupling_train"

echo "Auditing final SWA checkpoint $swa"
"$audit_python" -u "$repo/scripts/audit_flow_derivative_losses.py" \
  --model "$swa" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --coupling-target "$coupling_validation" \
  --output "$audit" --device cuda --delta 0.02 --batch-size 2048 \
  --sample-rows 200000 --sample-draws 64 --sample-seed 7301

date
echo "PLAIN_COMPLETE_FLOW_RESPONSE_DONE weight=$weight model=$swa audit=$audit"
