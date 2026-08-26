#!/usr/bin/env bash
# Archived GalSBI-prior job.
# Summarize the powered g1/g2 likelihood closure after both arrays pass.

#SBATCH --job-name=sbsi_galsbi50_sum
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1/logs/summary_%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1/logs/summary_%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1/bias_rblend_v1

for component in g1 g2; do
  mapfile -t results < <(find "$root/$component" -type f -name result.json | sort)
  if [[ "${#results[@]}" -ne 15 ]]; then
    echo "$component: expected 15 result files, found ${#results[@]}" >&2
    exit 2
  fi
  output="$root/${component}_summary.json"
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite $output" >&2
    exit 2
  fi
  "$python" "$repo/scripts/summarize_catalogue_bias.py" \
    --n-draws 32768 --output "$output" "${results[@]}"
done
