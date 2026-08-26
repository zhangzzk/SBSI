#!/usr/bin/env bash
# Generate three independent GalSBI catalogues and merge >=1M usable properties.

#SBATCH --job-name=sbsi_s5_bank
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
galsbi_repo=/home/z/Zekang.Zhang/galsbi
blendsim_repo=/home/z/Zekang.Zhang/blendsim
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
catalogue_root=/project/ls-gruen/users/zekang.zhang/cats/galsbi
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1

export PYTHONPATH="$repo:$galsbi_repo/src:$blendsim_repo"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

manifests=()
for seed in 20260822 20260823 20260824; do
  prefix="$catalogue_root/sbsi_prior_f24_seed${seed}"
  manifest="${prefix}_manifest.json"
  catalogue="${prefix}_0_r_ucat.gal.cat"
  if [[ ! -f "$manifest" || ! -f "$catalogue" ]]; then
    "$python" "$repo/scripts/generate_galsbi_prior_base.py" \
      --output-prefix "$prefix" --model Fischbacher+24 \
      --model-index 0 --seed "$seed" --galsbi-repository "$galsbi_repo"
  fi
  manifests+=(--catalogue-manifest "$manifest")
done

"$python" "$repo/scripts/build_galsbi_property_bank.py" \
  "${manifests[@]}" --blendsim-repository "$blendsim_repo" \
  --model Fischbacher+24 --model-index 0 --min-usable 1000000 \
  --output "$root/property_bank.parquet" \
  --manifest "$root/property_bank_manifest.json"
