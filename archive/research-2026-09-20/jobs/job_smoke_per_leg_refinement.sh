#!/usr/bin/env bash
#SBATCH --job-name=sbsi_per_leg_refine_smoke
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/flow_shape_augmentation_20260918_v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
"$python" -m pytest tests/test_guard_selection_mass.py tests/test_per_leg_refinement_selection.py tests/test_per_leg_guard_response.py tests/test_flow_guard_response.py -q
exec "$python" -u -m scripts.refine_per_leg_guard_flow --device cpu --smoke \
  --domain-root "$cache/harmonized_secondary_20260917_v1/domain" \
  --flow "$cache/harmonized_secondary_20260917_v1/refinement/selected.pt" \
  --reference-root "$root/per_leg_cv_smoke" --target-bank-root "$root" \
  --output-root "$root/per_leg_refinement_mass_smoke_v2"
