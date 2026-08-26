#!/usr/bin/env bash
# Archived GalSBI-prior job.
# Build a 50-case randomized-position scene from the fresh GalSBI base.

#SBATCH --job-name=sbsi_galsbi50
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendsim=/home/z/Zekang.Zhang/blendsim
galsbi=/home/z/Zekang.Zhang/galsbi
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1
base_prefix=/project/ls-gruen/users/zekang.zhang/cats/galsbi/sbsi_prior_f24_seed20260821

export PYTHONPATH="$repo:$blendsim:$galsbi/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

if [[ -e "$root/cases/gals0_0.0.feather" || -e "$root/scene_cases0_49.parquet" || -e "$root/scene_store" ]]; then
  echo "refusing to overwrite an existing 50-case GalSBI artifact" >&2
  exit 2
fi

"$python" "$blendsim/scripts/run_pipeline.py" \
  --config "$repo/configs/catalogue_prior_fresh_galsbi_50cases_v1.yaml" --steps 1

"$python" "$repo/scripts/assemble_galsbi_scene_catalogue.py" \
  --cases-directory "$root/cases" --case-start 0 --n-cases 50 \
  --shear-label 0.0 --base-manifest "${base_prefix}_manifest.json" \
  --output "$root/scene_cases0_49.parquet" --report "$root/scene_cases0_49_report.json" \
  --primary-mag-min 18 --primary-mag-max 25.8 \
  --primary-re-min 0.5 --primary-re-max 1.5

"$python" "$repo/scripts/build_scene_prior.py" \
  --catalogue "$root/scene_cases0_49.parquet" --output "$root/scene_store" \
  --guard-radius-arcsec 11 --weight-column prior_weight --group-column case
