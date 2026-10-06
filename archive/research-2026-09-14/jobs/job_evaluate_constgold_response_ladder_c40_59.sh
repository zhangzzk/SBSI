#!/usr/bin/env bash
# Twenty-case ConstGold response ladder: fixed populations, detection, complete.

#SBATCH --job-name=sbsi_cg_Rladder20
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_ladder_c40_59_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_ladder_c40_59_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python}
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
constroot=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_ladder_c40_59_v1
output=$root/result.json

mkdir -p "$root/logs"
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

models=()
for seed in 501 502 503 504; do
  path=$cache/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s${seed}_swaavg.pt
  [[ -s "$path" ]] || { echo "missing measurement model $path" >&2; exit 2; }
  models+=(--measurement-model "$path")
done
cases=()
for case in $(seq 40 59); do
  cases+=(--case "$case")
done

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
"$python" -u -m scripts.evaluate_constgold_complete_forward_response \
  --constgold-root "$constroot" \
  --crowd-lookup "$cache/crowd_flux_conc_c0-199.feather" \
  --rblend-lookup "$cache/mixed_shear_cde/constgold/rblend_v3_c40-49.feather" \
  --rblend-lookup "$cache/mixed_shear_cde/constgold/rblend_v3_c50-59.feather" \
  "${models[@]}" "${cases[@]}" \
  --emulator-model "$repo/models/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json" \
  --emulator-metadata "$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json" \
  --output "$output" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 2048 \
  --n-boot 10000 --bootstrap-seed 20260902 --device cuda

echo "CONSTGOLD_RESPONSE_LADDER20_DONE output=$output"
