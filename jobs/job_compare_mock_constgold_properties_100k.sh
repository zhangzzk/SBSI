#!/usr/bin/env bash
# Compare 100k selected complete-likelihood mock measurements with 100k
# selected ConstGold +g measurements.  The source catalogue is 9.4 GB, so the
# scan belongs on a scheduler node even though the final plot is small.

#SBATCH --job-name=sbsi_mock_constgold_100k
#SBATCH --partition=inter
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1
mock=$root/prepare_g002/mock
output=$root/mock_constgold_comparison_100k_v3
constgold=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

"$python" "$repo/scripts/compare_mock_constgold_properties.py" \
  --mock "$mock/measurements.parquet" \
  --mock-manifest "$mock/likelihood_mock_manifest.json" \
  --constgold "$constgold" \
  --output "$output" \
  --sample-size 100000 \
  --seed 20260901 \
  --min-case 40 \
  --pixel-size 0.2 \
  --shape-max 0.6 \
  --mag-max 25.8 \
  --radius-min 0.75
