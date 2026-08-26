#!/usr/bin/env bash
# Archived GalSBI-prior job.
# Resource/log wrapper for a 50-case fresh-prior closure profile.

#SBATCH --job-name=sbsi_galsbi50_prof
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1/logs/profile_%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1/logs/profile_%x_%j.err

set -euo pipefail

env \
  PRIOR_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_50cases_v1 \
  RUN_NAME="${RUN_NAME:-closure_g1_v1}" \
  PROFILE="${PROFILE:-0.005,0.01,0.015,0.02,0.025,0.03,0.035}" \
  LADDER="${LADDER:-8192,32768}" \
  PROPOSAL_SEEDS="${PROPOSAL_SEEDS:-6701,6702}" \
  N_DETECTED="${N_DETECTED:-1024}" \
  bash /home/z/Zekang.Zhang/SBSI/jobs/job_fresh_galsbi_closure_profile.sh
