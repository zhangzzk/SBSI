#!/usr/bin/env bash
# Pilot: add one paired-transport loss to the 100/100 flow; no inference correction.

#SBATCH --job-name=sbsi_flow_pair1
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/paired_transport_train_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/paired_transport_train_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
train_python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
audit_python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
base=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1
root=$base/paired_transport_train_v1

g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$source_root/det_meas_crowd_conc_g0.05_all200.feather
response=$base/response_target_mixed_train160_6x6x5_matrix_re037.npz
coupling_train=$base/coupling_target_mixed_train160_6x6x5_re037.npz
coupling_validation=$base/coupling_target_mixed_validation40_6x6x5_re037.npz
parent=$base/r100_t100_v1/measurement_flow_plain_direct_r100_t100_d16_re037_s501_swaavg.pt
output=$root/measurement_flow_paired1_r100_t100_full_re037_s501_swabase.pt
swa=${output/swabase/swaavg}
audit=$root/complete_response_paired1_best_re037_s501.json
gap=$root/response_loss_gap_paired1_best_re037_s501_200k.json

mkdir -p "$root/logs"
for path in "$g0" "$g05" "$response" "$coupling_train" "$coupling_validation" "$parent"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
for path in "$output" "$swa" "${output%.pt}_train_curve.npz" "$audit" "$gap"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

export PYTHONPATH=$repo
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
unset PYTORCH_CUDA_ALLOC_CONF

date
nvidia-smi -L
echo "Fine-tuning 100/100 with one paired forward-transport MSE; all selected rows; five-epoch pilot"
"$train_python" -u -m sbsi.flow_training \
  --catalogue "$g0" --additional-catalogue "$g05" --output "$output" \
  --target-column detected --selection-name sextractor_detected \
  --feature-set g0_meas_crowd_conc_szfl_noz \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type affine \
  --fold-catalogue-shear --max-cases 200 --expected-case-count 200 \
  --validation-group-column case --validation-size 0.2 \
  --max-rows 0 --epochs 5 --batch-size 4096 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 \
  --lr 0.00001 --patience 8 --weight-decay 0 --swa-last-k 5 \
  --seed 501 --num-workers 8 --gpu-resident \
  --primary-mag-max 25.8 --primary-re-min 0.37 \
  --fine-tune-reference "$parent" --fine-tune-reuse-transforms --nll-weight 1 --preserve-weight 0 \
  --paired-transport-weight 1 --paired-transport-shear 0.05 \
  --response-estimator complete_flow --response-draws 16 --response-sample-seed 7301 \
  --response-weight 100 --response-delta 0.02 \
  --response-difference central --response-target-npz "$response" \
  --response-components matrix --response-error absolute \
  --response-rel-floor 0.05 --response-bin-ema 0 \
  --coupling-weight 100 --coupling-target-npz "$coupling_train"

# Primary candidate: best held-out NLL + existing responses + paired-transport objective.
echo "Auditing density and derivative losses with the same held-out protocol"
"$audit_python" -u "$repo/scripts/audit_flow_derivative_losses.py" \
  --model "$output" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --coupling-target "$coupling_validation" \
  --output "$audit" --device cuda --delta 0.02 --batch-size 2048 \
  --sample-rows 200000 --sample-draws 64 --sample-seed 7301
"$audit_python" -u "$repo/scripts/diagnose_flow_response_loss_gap.py" \
  --model "$output" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --output "$gap" \
  --device cuda --delta 0.02 --batch-size 1024 --sample-rows 200000 \
  --sample-seed 8301 --qmc-draws 64 --antithetic-draws

sha256sum "$swa" "$audit" "$gap"
date
echo "PAIRED_FLOW_PILOT_DONE model=$output audit=$audit gap=$gap"
