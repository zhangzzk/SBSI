#!/usr/bin/env bash
# Build a complete cases-0--9 FS2 empirical prior from existing truth catalogues.

#SBATCH --job-name=sbsi_fs2_scene
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_full_cases0_9_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_full_cases0_9_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
: "${FS2_INPUT:=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876}"
: "${OUTPUT_ROOT:=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_full_cases0_9_v1}"

catalogue="$OUTPUT_ROOT/scene_catalogue.parquet"
report="$OUTPUT_ROOT/scene_catalogue_report.json"
scene_store="$OUTPUT_ROOT/scene_store"
for target in "$catalogue" "$report" "$scene_store"; do
  if [[ -e "$target" ]]; then
    echo "refusing to overwrite $target" >&2
    exit 2
  fi
done

export PYTHONPATH="$repo"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/assemble_fs2_scene_catalogue.py" \
  --cases-directory "$FS2_INPUT" --cases 0-9 --shear-label 0.0 \
  --expected-g1 0 --expected-g2 0 \
  --output "$catalogue" --report "$report" \
  --primary-mag-min 18 --primary-mag-max 25.8 \
  --primary-re-min 0.5 --primary-re-max 1.5 \
  --expected-rows 6995680 --expected-positive-atoms 637226

"$python" "$repo/scripts/build_scene_prior.py" \
  --catalogue "$catalogue" --output "$scene_store" \
  --guard-radius-arcsec 11 --weight-column prior_weight --group-column case \
  --provenance-manifest "$report"
