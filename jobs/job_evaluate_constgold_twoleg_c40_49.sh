#!/usr/bin/env bash
# Two-leg ConstGold response on cases 40--49: matched both-detected identities
# and unmatched per-leg detected means, measured versus flow+R_blend.

#SBATCH --job-name=sbsi_cg_2leg10
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_twoleg_c40_49_v4/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_twoleg_c40_49_v4/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python}
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
constroot=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_twoleg_c40_49_v4
output=$root/result.json

mkdir -p "$root/logs"
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

models=()
for seed in 501 502 503 504; do
  path=$cache/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s${seed}_swaavg.pt
  [[ -s "$path" ]] || { echo "missing measurement model $path" >&2; exit 2; }
  models+=(--measurement-model "$path")
done

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$repo"
"$python" -u -m scripts.evaluate_constgold_twoleg_response \
  --plus-catalogue "$constroot/constant_shear_catalogue_0.02_train.feather" \
  --minus-catalogue "$constroot/constant_shear_catalogue_-0.02_train.feather" \
  --crowd-lookup "$cache/crowd_flux_conc_c0-199.feather" \
  --rblend-lookup "$cache/mixed_shear_cde/constgold/rblend_v3_c40-49.feather" \
  "${models[@]}" \
  --output "$output" \
  --min-case 40 --max-case 50 --h 0.02 \
  --draws 64 --sampling-seed 7301 \
  --batch-size 2048 --n-boot 10000 --bootstrap-seed 20260901 \
  --device cuda

echo "CONSTGOLD_TWOLEG_DONE output=$output"
