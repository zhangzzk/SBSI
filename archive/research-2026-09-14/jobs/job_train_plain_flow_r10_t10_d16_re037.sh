#!/usr/bin/env bash
# Train the new-domain pure-NLL parent, then the balanced 10/10 direct-response flow.

#SBATCH --job-name=sbsi_Er10R037
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
train_python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
audit_python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1

g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$source_root/det_meas_crowd_conc_g0.05_all200.feather
response=$root/response_target_mixed_train160_6x6x5_matrix_re037.npz
coupling_train=$root/coupling_target_mixed_train160_6x6x5_re037.npz
coupling_validation=$root/coupling_target_mixed_validation40_6x6x5_re037.npz
parent=$root/measurement_flow_plain_pure_nll_re037_s501_swabase.pt
parent_swa=${parent/swabase/swaavg}
output=$root/measurement_flow_plain_direct_r10_t10_d16_re037_s501_swabase.pt
swa=${output/swabase/swaavg}
audit=$root/complete_response_plain_direct_r10_t10_d16_re037_s501.json
gap=$root/response_loss_gap_r10_t10_d16_re037_s501_200k.json

mkdir -p "$root/logs"
for path in "$g0" "$g05" "$response" "$coupling_train" "$coupling_validation"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
for path in \
  "$parent" "$parent_swa" "${parent%.pt}_train_curve.npz" \
  "$output" "$swa" "${output%.pt}_train_curve.npz" "$audit" "$gap"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

export PYTHONPATH=$repo
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
unset PYTORCH_CUDA_ALLOC_CONF

date
nvidia-smi -L
echo "Stage 1/3: training the Re>0.37 pure-NLL parent from scratch"
"$train_python" -u -m sbsi.flow_training \
  --catalogue "$g0" --additional-catalogue "$g05" --output "$parent" \
  --target-column detected --selection-name sextractor_detected \
  --feature-set g0_meas_crowd_conc_szfl_noz \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type affine \
  --fold-catalogue-shear --max-cases 200 --expected-case-count 200 \
  --validation-group-column case --validation-size 0.2 \
  --max-rows 4000000 --epochs 80 --batch-size 8192 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 \
  --lr 0.0007 --patience 10 --weight-decay 1e-5 --swa-last-k 8 \
  --seed 501 --num-workers 8 --gpu-resident \
  --primary-mag-max 25.8 --primary-re-min 0.37

echo "Stage 2/3: fine-tuning with 10*L_R + 10*L_theta and 16 antithetic draws"
"$train_python" -u -m sbsi.flow_training \
  --catalogue "$g0" --additional-catalogue "$g05" --output "$output" \
  --target-column detected --selection-name sextractor_detected \
  --feature-set g0_meas_crowd_conc_szfl_noz \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type affine \
  --fold-catalogue-shear --max-cases 200 --expected-case-count 200 \
  --validation-group-column case --validation-size 0.2 \
  --max-rows 4000000 --epochs 30 --batch-size 4096 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 \
  --lr 0.0001 --patience 8 --weight-decay 0 --swa-last-k 5 \
  --seed 501 --num-workers 8 --gpu-resident \
  --primary-mag-max 25.8 --primary-re-min 0.37 \
  --fine-tune-reference "$parent_swa" --nll-weight 1 --preserve-weight 0 \
  --response-estimator complete_flow --response-draws 16 --response-sample-seed 7301 \
  --response-weight 10 --response-delta 0.02 \
  --response-difference central --response-target-npz "$response" \
  --response-components matrix --response-error absolute \
  --response-rel-floor 0.05 --response-bin-ema 0 \
  --coupling-weight 10 --coupling-target-npz "$coupling_train"

echo "Stage 3/3: auditing density, derivatives, and controlled response loss"
"$audit_python" -u "$repo/scripts/audit_flow_derivative_losses.py" \
  --model "$swa" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --coupling-target "$coupling_validation" \
  --output "$audit" --device cuda --delta 0.02 --batch-size 2048 \
  --sample-rows 200000 --sample-draws 64 --sample-seed 7301
"$audit_python" -u "$repo/scripts/diagnose_flow_response_loss_gap.py" \
  --model "$swa" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --output "$gap" \
  --device cuda --delta 0.02 --batch-size 1024 --sample-rows 200000 \
  --sample-seed 8301 --qmc-draws 64 --antithetic-draws

date
echo "PLAIN_FLOW_R10_T10_D16_RE037_DONE model=$swa audit=$audit gap=$gap"
