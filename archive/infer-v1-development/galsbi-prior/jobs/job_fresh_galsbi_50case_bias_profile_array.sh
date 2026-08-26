#!/usr/bin/env bash
# Archived GalSBI-prior job.
# Resource/log wrapper for the powered 50-case fresh-prior calibration.

#SBATCH --job-name=sbsi_galsbi50_bias
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-14%4
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1/logs/bias_%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1/logs/bias_%x_%A_%a.err

set -euo pipefail

env \
  PRIOR_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1 \
  COMPONENT="${COMPONENT:-g1}" \
  CALIBRATION_NAME="${CALIBRATION_NAME:-bias_rblend_v1}" \
  bash /home/z/Zekang.Zhang/SBSI/jobs/job_fresh_galsbi_bias_profile_array.sh
