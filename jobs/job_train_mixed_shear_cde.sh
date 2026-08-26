#!/usr/bin/env bash
# Train one of the controlled mixed-shear C/D/E arms (array index 0/1/2).

#SBATCH --job-name=sbsi_mixcde
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde

g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$root/det_meas_crowd_conc_g0.05_all200.feather
target=$root/response_target_mixed_train160_6x6x5_matrix.npz

case "${SLURM_ARRAY_TASK_ID:?submit as array 0-2}" in
  0) arm=C; response=0; components=trace ;;
  1) arm=D; response=450; components=trace ;;
  2) arm=E; response=450; components=matrix ;;
  *) echo "unknown array arm $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

output=$root/measurement_flow_mixed_g0_g005_${arm}_s501_swabase.pt
export PYTHONPATH=$repo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
unset PYTORCH_CUDA_ALLOC_CONF

for path in "$g0" "$g05"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
if (( response > 0 )); then
  [[ -f "$target" ]] || { echo "missing response target: $target" >&2; exit 2; }
fi
for path in "$output" "${output/swabase/swaavg}" "${output%.pt}_train_curve.npz"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done

response_args=()
if (( response > 0 )); then
  response_args=(
    --response-weight "$response" --response-delta 0.02
    --response-difference central --response-target-npz "$target"
    --response-components "$components" --response-error absolute
    --response-rel-floor 0.05 --response-bin-ema 0.0
  )
fi

echo "Training arm=$arm response_weight=$response components=$components seed=501"
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
  --coupling-weight 0 "${response_args[@]}"
date
echo "MIXED_SHEAR_TRAIN_DONE arm=$arm output=$output"
