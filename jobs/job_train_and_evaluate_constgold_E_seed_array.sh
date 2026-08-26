#!/usr/bin/env bash
# Train three additional replicas of original mixed-shear arm E, then score ConstGold.

#SBATCH --job-name=sbsi_e_seeds
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde
constgold_root=$root/constgold

seeds=(502 503 504)
seed=${seeds[${SLURM_ARRAY_TASK_ID:?submit as array 0-2}]}
g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05=$root/det_meas_crowd_conc_g0.05_all200.feather
target=$root/response_target_mixed_train160_6x6x5_matrix.npz
output=$root/measurement_flow_mixed_g0_g005_E_s${seed}_swabase.pt
model=${output/swabase/swaavg}

catalogue=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
crowd=/project/ls-gruen/users/zekang.zhang/sbsi_caches/crowd_flux_conc_c0-199.feather
score=$constgold_root/constgold_response_E_s${seed}_c40-139.json

for path in "$g0" "$g05" "$target" "$catalogue" "$crowd"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
for path in "$output" "$model" "${output%.pt}_train_curve.npz" "$score"; do
  [[ ! -e "$path" ]] || { echo "refusing to overwrite $path" >&2; exit 2; }
done
lookups=()
for first in $(seq 40 10 130); do
  last=$((first + 9))
  path=$constgold_root/rblend_v3_c${first}-${last}.feather
  [[ -s "$path" ]] || { echo "missing lookup shard: $path" >&2; exit 2; }
  lookups+=("$path")
done

export PYTHONPATH=$repo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
unset PYTORCH_CUDA_ALLOC_CONF

echo "Training original arm E replica seed=$seed"
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
  --seed "$seed" --num-workers 8 --gpu-resident \
  --primary-mag-max 25.8 --primary-re-min 0.5 \
  --coupling-weight 0 \
  --response-weight 450 --response-delta 0.02 \
  --response-difference central --response-target-npz "$target" \
  --response-components matrix --response-error absolute \
  --response-rel-floor 0.05 --response-bin-ema 0.0

[[ -s "$model" ]] || { echo "training did not produce $model" >&2; exit 2; }
echo "Evaluating original arm E replica seed=$seed on ConstGold"
"$python" -u "$repo/scripts/evaluate_constgold_response.py" \
  --model "$model" --catalogue "$catalogue" \
  --blend-lookup "${lookups[@]}" --crowd-lookup "$crowd" \
  --output "$score" --min-case 40 --max-case 140 \
  --batch-size 65536 --n-boot 10000 --bootstrap-seed 0 --device cuda
date
echo "E_SEED_CONSTGOLD_DONE seed=$seed model=$model score=$score"
