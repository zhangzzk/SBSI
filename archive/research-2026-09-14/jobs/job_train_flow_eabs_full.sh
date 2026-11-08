#!/usr/bin/env bash
# Intrinsic-|e| photometric target with uncapped training, warm-started from the same pure-NLL parent.

#SBATCH --job-name=sbsi_Eeabs2full
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/eabs2_full_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/eabs2_full_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
train_python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
audit_python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
base=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1
root=$base/eabs2_full_v1

g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$source_root/det_meas_crowd_conc_g0.05_all200.feather
response=$base/response_target_mixed_train160_6x6x5_matrix_re037.npz
coupling_train=$root/coupling_train_eabs2.npz
coupling_validation=$root/coupling_validation_eabs2.npz
parent=$base/measurement_flow_plain_pure_nll_re037_s501_swaavg.pt
output=$root/measurement_flow_plain_eabs2_r100_t100_full_re037_s501_swabase.pt
swa=${output/swabase/swaavg}
audit=$root/complete_response_eabs2_full_re037_s501.json
gap=$root/response_loss_gap_eabs2_full_re037_s501_allval.json

mkdir -p "$root/logs"
for path in "$g0" "$g05" "$response" "$coupling_train" "$coupling_validation" "$parent"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
for path in "$output" "$swa" "${output%.pt}_train_curve.npz" "$audit" "$gap"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
unset PYTORCH_CUDA_ALLOC_CONF

cd "$repo"
date
nvidia-smi -L
"$audit_python" - "$root/target_audit.json" <<'PYGATE'
import json, sys
with open(sys.argv[1]) as f:
    audit = json.load(f)
if audit.get('status') != 'pass':
    raise SystemExit('target audit did not pass')
PYGATE
echo "Uncapped fine-tune: 100*L_R + 100*L_theta on the 6x6x5x2 intrinsic-|e| target"
"$train_python" -u -m sbsi.flow_training \
  --catalogue "$g0" --additional-catalogue "$g05" --output "$output" \
  --target-column detected --selection-name sextractor_detected \
  --feature-set g0_meas_crowd_conc_szfl_noz \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type affine \
  --fold-catalogue-shear --max-cases 200 --expected-case-count 200 \
  --validation-group-column case --validation-size 0.2 \
  --max-rows 0 --epochs 30 --batch-size 4096 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 \
  --lr 0.0001 --patience 8 --weight-decay 0 --swa-last-k 5 \
  --seed 501 --num-workers 8 --gpu-resident \
  --primary-mag-max 25.8 --primary-re-min 0.37 \
  --fine-tune-reference "$parent" --fine-tune-reuse-transforms --nll-weight 1 --preserve-weight 0 \
  --response-estimator complete_flow --response-draws 16 --response-sample-seed 7301 \
  --response-weight 100 --response-delta 0.02 \
  --response-difference central --response-target-npz "$response" \
  --response-components matrix --response-error absolute \
  --response-rel-floor 0.05 --response-bin-ema 0 \
  --coupling-weight 100 --coupling-target-npz "$coupling_train"

echo "Auditing density and derivative losses with the same held-out protocol"
"$audit_python" -u -m scripts.audit_flow_derivative_losses \
  --model "$swa" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --coupling-target "$coupling_validation" \
  --output "$audit" --device cuda --delta 0.02 --batch-size 2048 \
  --sample-rows 2000000 --sample-draws 64 --sample-seed 7301
"$audit_python" -u -m scripts.diagnose_flow_response_loss_gap \
  --model "$swa" --g0-catalogue "$g0" --sheared-catalogue "$g05" \
  --response-target "$response" --output "$gap" \
  --device cuda --delta 0.02 --batch-size 1024 --sample-rows 2000000 \
  --sample-seed 8301 --qmc-draws 64 --antithetic-draws

sha256sum "$swa" "$audit" "$gap"
date
echo "FLOW_EABS2_FULL_DONE model=$swa audit=$audit gap=$gap"
