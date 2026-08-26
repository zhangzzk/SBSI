#!/usr/bin/env bash
# Build twenty independent ten-case scene stores from the fresh FS2 prior.

#SBATCH --job-name=sbsi_fs2p_scene
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-19%4
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_prior_cases20000_20199_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_prior_cases20000_20199_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_prior_cases20000_20199_v1
: "${SLURM_ARRAY_TASK_ID:?this wrapper requires a Slurm array task}"

lower=$((20000 + 10 * SLURM_ARRAY_TASK_ID))
upper=$((lower + 9))
shard="$root/shards/cases${lower}_${upper}"
catalogue="$shard/scene_catalogue.parquet"
report="$shard/scene_catalogue_report.json"
scene_store="$shard/scene_store"

for target in "$shard" "$catalogue" "$report" "$scene_store"; do
  if [[ -e "$target" ]]; then
    echo "refusing to overwrite $target" >&2
    exit 2
  fi
done
if [[ ! -f "$root/generation_manifest.json" ]]; then
  echo "missing generation manifest" >&2
  exit 2
fi

export PYTHONPATH="$repo"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/assemble_fs2_scene_catalogue.py" \
  --cases-directory "$root/cases" --cases "$lower-$upper" --shear-label 0.0 \
  --expected-g1 0 --expected-g2 0 \
  --output "$catalogue" --report "$report" \
  --primary-mag-min 18 --primary-mag-max 25.8 \
  --primary-re-min 0.5 --primary-re-max 1.5 \
  --expected-rows 6996800

"$python" "$repo/scripts/build_scene_prior.py" \
  --catalogue "$catalogue" --output "$scene_store" \
  --guard-radius-arcsec 11 --weight-column prior_weight --group-column case \
  --provenance-manifest "$root/generation_manifest.json" \
  --provenance-manifest "$report"
