#!/usr/bin/env bash
# Draw 100 fixed-density cases and cache their 11-arcsec neighbour graph.

#SBATCH --job-name=sbsi_s5_scene
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendsim_repo=/home/z/Zekang.Zhang/blendsim
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1

export PYTHONPATH="$repo:$blendsim_repo"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/randomize_galsbi_scene_cases.py" \
  --property-bank "$root/property_bank.parquet" \
  --property-manifest "$root/property_bank_manifest.json" \
  --blendsim-repository "$blendsim_repo" \
  --output "$root/scene_cases0_99.parquet" \
  --report "$root/scene_cases0_99_report.json" \
  --n-cases 100 --case-start 0 --rows-per-case 71104 \
  --primary-mag-min 18 --primary-mag-max 25.8 \
  --primary-re-min 0.5 --primary-re-max 1.5

"$python" "$repo/scripts/build_scene_prior.py" \
  --catalogue "$root/scene_cases0_99.parquet" \
  --output "$root/scene_store" --guard-radius-arcsec 11 \
  --weight-column prior_weight --group-column case \
  --provenance-manifest "$root/property_bank_manifest.json" \
  --provenance-manifest "$root/scene_cases0_99_report.json"
