#!/usr/bin/env bash
# Archived GalSBI-prior job.
# Fresh Fischbacher+24 model-index-0 population for disjoint prior cases.

#SBATCH --job-name=sbsi_galsbi_base
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
galsbi_repo=/home/z/Zekang.Zhang/galsbi
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
output_prefix=/project/ls-gruen/users/zekang.zhang/cats/galsbi/sbsi_prior_f24_seed20260821

export PYTHONPATH="$repo:$galsbi_repo/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/generate_galsbi_prior_base.py" \
  --output-prefix "$output_prefix" \
  --model Fischbacher+24 --model-index 0 --seed 20260821 \
  --galsbi-repository "$galsbi_repo"
