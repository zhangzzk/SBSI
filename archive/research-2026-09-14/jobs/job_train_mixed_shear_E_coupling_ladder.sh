#!/usr/bin/env bash
# Density-preserving Flow-E screen over photometric coupling strength.

#SBATCH --job-name=sbsi_Etheta
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-7%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
source_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
target_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_photometric
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_e_coupling_ladder_v1

case "${SLURM_ARRAY_TASK_ID:?submit as array 0-7}" in
  0) coupling_weight=0.5; tag=t0p5 ;;
  1) coupling_weight=1; tag=t1 ;;
  2) coupling_weight=2; tag=t2 ;;
  3) coupling_weight=5; tag=t5 ;;
  4) coupling_weight=10; tag=t10 ;;
  5) coupling_weight=25; tag=t25 ;;
  6) coupling_weight=75; tag=t75 ;;
  7) coupling_weight=200; tag=t200 ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$source_root/det_meas_crowd_conc_g0.05_all200.feather
response=$source_root/response_target_mixed_train160_6x6x5_matrix.npz
coupling=$target_root/coupling_target_mixed_train160_6x6x5.npz
output=$root/measurement_flow_mixed_g0_g005_E_r450_${tag}_s501_swabase.pt

export PYTHONPATH=$repo
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "$root/logs"
for path in "$g0" "$g05" "$response" "$coupling"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
for path in "$output" "${output/swabase/swaavg}" "${output%.pt}_train_curve.npz"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

echo "Training Flow-E response_weight=450 coupling_weight=$coupling_weight seed=501"
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
  --coupling-weight "$coupling_weight" --coupling-target-npz "$coupling" \
  --response-weight 450 --response-delta 0.02 \
  --response-difference central --response-target-npz "$response" \
  --response-components matrix --response-error absolute \
  --response-rel-floor 0.05 --response-bin-ema 0.0

date
echo "FLOW_E_COUPLING_LADDER_TRAIN_DONE tag=$tag output=$output"
