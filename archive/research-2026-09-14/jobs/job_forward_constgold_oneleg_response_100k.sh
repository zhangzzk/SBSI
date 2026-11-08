#!/usr/bin/env bash
# Forward numerical response of the complete conditional measurement model on
# the frozen 100k ConstGold plus-selected identities.  One GPU is sufficient:
# this is 4 x 64 paired flow draws per truth context, not a 12.76M-atom inverse
# likelihood scan.

#SBATCH --job-name=sbsi_cg_fwdR_100k
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_oneleg_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_oneleg_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python}
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_oneleg_v1
sample=${SAMPLE:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1/mock_constgold_comparison_100k_v3/constgold_plus_selected_sample_100k.parquet}
constgold=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
crowd=$cache/crowd_flux_conc_c0-199.feather
model=$cache/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
output=$root/${OUTPUT_TAG:-result}
draws=${DRAWS:-64}
draw_ladder=${DRAW_LADDER:-16,32,64}
sampling_seeds=${SAMPLING_SEEDS:-4101,4102,4103,4104}

mkdir -p "$root/logs"
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

rblend=()
for first in $(seq 40 10 130); do
  last=$((first + 9))
  path=$cache/mixed_shear_cde/constgold/rblend_v3_c${first}-${last}.feather
  [[ -s "$path" ]] || { echo "missing R_blend shard $path" >&2; exit 2; }
  rblend+=(--rblend-lookup "$path")
done

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
extra_args=()
if [[ -n "${MAX_OBJECTS:-}" ]]; then
  extra_args+=(--max-objects "$MAX_OBJECTS")
fi
"$python" -u "$repo/scripts/forward_constgold_oneleg_response.py" \
  --sample "$sample" \
  --constgold "$constgold" \
  --crowd-lookup "$crowd" \
  "${rblend[@]}" \
  --measurement-model "$model" \
  --output "$output" \
  --h 0.02 \
  --draws "$draws" \
  --draw-ladder "$draw_ladder" \
  --sampling-seeds "$sampling_seeds" \
  --batch-size 2048 \
  --n-boot 10000 \
  --bootstrap-seed 20260901 \
  --device cuda \
  "${extra_args[@]}"

echo "FORWARD_CONSTGOLD_ONELEG_DONE output=$output"
