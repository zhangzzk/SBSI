#!/usr/bin/env bash
# Archived GalSBI-prior job.
# Resource/log wrapper for the 50-case fresh-prior response cache.

#SBATCH --job-name=sbsi_galsbi50_rb
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1/logs/rblend_%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1/logs/rblend_%x_%j.err

set -euo pipefail

env \
  PRIOR_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1 \
  bash /home/z/Zekang.Zhang/SBSI/jobs/job_build_fresh_galsbi_blend_response.sh
