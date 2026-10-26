#!/usr/bin/env bash
# Mean-head-only response fine-tuning from the pure-NLL Flow-E endpoint.

#SBATCH --job-name=sbsi_Epres
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-4%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_preserve_finetune_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_preserve_finetune_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
train_python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
audit_python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
target_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric
ladder_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_equal_ladder_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_preserve_finetune_v1

case "${SLURM_ARRAY_TASK_ID:?submit as array 0-4}" in
  0) beta=0 ;;
  1) beta=1 ;;
  2) beta=0.03 ;;
  3) beta=0.1 ;;
  4) beta=0.3 ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

tag="beta${beta}"
g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$source_root/det_meas_crowd_conc_g0.05_all200.feather
response=$source_root/response_target_mixed_train160_6x6x5_matrix.npz
coupling_train=$target_root/coupling_target_mixed_train160_6x6x5.npz
coupling_validation=$target_root/coupling_target_mixed_validation40_6x6x5.npz
reference=$ladder_root/measurement_flow_mixed_g0_g005_E_r0_t0_s501_swaavg.pt
output=$root/measurement_flow_mixed_g0_g005_E_preserve_${tag}_s501_swabase.pt
swa=${output/swabase/swaavg}
audit=$root/heldout_derivative_losses_E_preserve_${tag}_s501.json

export PYTHONPATH=$repo
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "$root/logs"
for path in "$g0" "$g05" "$response" "$coupling_train" "$coupling_validation" "$reference"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
for path in "$output" "$swa" "${output%.pt}_train_curve.npz" "$audit"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

echo "Fine-tuning Flow-E mean head beta=$beta from pure-NLL reference, seed=501"
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
  --max-rows 4000000 --epochs 60 --batch-size 8192 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 \
  --lr 0.0001 --patience 10 --weight-decay 0 --swa-last-k 5 \
  --seed 501 --num-workers 8 --gpu-resident \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --coupling-weight 1 --coupling-target-npz "$coupling_train" \
  --response-weight 1 --response-delta 0.02 \
  --response-difference central --response-target-npz "$response" \
  --response-components matrix --response-error absolute \
  --response-rel-floor 0.05 --response-bin-ema 0.0 \
  --nll-weight 0 --fine-tune-reference "$reference" \
  --preserve-weight "$beta" --fine-tune-mean-only

echo "Auditing final SWA checkpoint $swa"
"$audit_python" -u "$repo/scripts/audit_flow_derivative_losses.py" \
  --model "$swa" --reference-model "$reference" \
  --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --coupling-target "$coupling_validation" \
  --output "$audit" --device cuda --delta 0.02

date
echo "FLOW_E_PRESERVE_FINETUNE_DONE tag=$tag model=$swa audit=$audit"
