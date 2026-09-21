#!/usr/bin/env bash
# Submit after successful harmonized density and reference audits.
#SBATCH --job-name=sbsi_harmonized_refine
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=38G
#SBATCH --time=01:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.err
set -euo pipefail
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/harmonized_secondary_20260917_v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
exec /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.refine_guard_control_variate_flow \
  --domain-root "$root/domain" \
  --flow "$cache/full_domain_true_mag26_guard_v1/flow/guard/selected.pt" \
  --reference-root "$root/guard_reference" \
  --output-root "$root/refinement"
