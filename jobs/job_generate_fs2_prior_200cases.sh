#!/usr/bin/env bash
# Generate 200 fresh unsheared FS2 prior cases and freeze their provenance.

#SBATCH --job-name=sbsi_fs2p_gen
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_prior_cases20000_20199_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_prior_cases20000_20199_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_prior_cases20000_20199_v1
cases="$root/cases"
manifest="$root/generation_manifest.json"
config="$repo/configs/catalogue_prior_fs2_200cases_v1.yaml"
base=/project/ls-gruen/users/zekang.zhang/lsst_sims/FS2/FS2_25876/FS2_25876_IA_Griffith_UDF_i_25.feather
: "${MODE:=generate}"

if [[ -e "$manifest" ]]; then
  echo "refusing to overwrite existing generation manifest $manifest" >&2
  exit 2
fi
if [[ "$MODE" == generate ]]; then
  if [[ -e "$cases" ]]; then
    echo "refusing to overwrite existing FS2 prior cases $cases" >&2
    exit 2
  fi
elif [[ "$MODE" == audit ]]; then
  if [[ ! -d "$cases" ]]; then
    echo "audit recovery requires existing FS2 prior cases $cases" >&2
    exit 2
  fi
else
  echo "MODE must be generate or audit" >&2
  exit 2
fi

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

if [[ "$MODE" == generate ]]; then
  "$python" "$blendemu/scripts/run_pipeline.py" --config "$config" --steps 1
fi

"$python" "$repo/scripts/audit_fs2_prior_generation.py" \
  --cases-directory "$cases" --cases 20000-20199 \
  --forbidden-cases 0-9999 --expected-rows-per-case 699680 \
  --seed-offset 123 --base-catalogue "$base" --pipeline-config "$config" \
  --generator "$blendemu/scripts/run_pipeline.py" \
  --generator-catalog-module "$blendemu/blendemu/catalog.py" \
  --output "$manifest"
