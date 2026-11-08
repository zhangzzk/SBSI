#!/usr/bin/env bash
# Prepare a second, identity-disjoint 100k ConstGold plus-selected sample.

#SBATCH --job-name=sbsi_cg_sample2_100k
#SBATCH --partition=inter
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_oneleg_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_oneleg_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_oneleg_v1
constgold=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
exclude=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1/mock_constgold_comparison_100k_v3/constgold_plus_selected_sample_100k.parquet
output=$root/replication_sample_100k

[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
"$python" -u "$repo/scripts/prepare_disjoint_constgold_plus_sample.py" \
  --constgold "$constgold" \
  --exclude-sample "$exclude" \
  --output "$output" \
  --sample-size 100000 \
  --seed 20260902 \
  --min-case 40 \
  --max-case 140 \
  --pixel-size 0.2 \
  --shape-max 0.6 \
  --mag-max 25.8 \
  --radius-min 0.75

echo "DISJOINT_CONSTGOLD_SAMPLE_DONE output=$output"
