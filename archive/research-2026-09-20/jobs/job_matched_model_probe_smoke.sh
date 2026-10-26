#!/usr/bin/env bash
#SBATCH --job-name=sbsi_matched_model_smoke
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
base=/project/ls-gruen/users/zekang.zhang
"$python" -m pytest tests/test_evaluate_matched_model_probe.py tests/test_matched_model_probe.py tests/test_matched_response_probe.py -q
exec "$python" -u -m scripts.evaluate_matched_model_probe \
  --probe-root "$base/blendemu_runs/random_sign_probe_pilot_20260918_v1/images" \
  --prepared-root "$base/sbsi_caches/fresh_fullscene_joint_20260918_v1/smoke" \
  --flow "initial=$base/sbsi_caches/harmonized_secondary_20260917_v1/refinement/selected.pt" \
  --flow "common_descent=$base/sbsi_caches/flow_shape_augmentation_20260918_v1/joint_probability_refinement_full/selected.pt" \
  --emulator "$base/blendemu_runs/disk_response_moment_20260918_v1/model" \
  --output "$base/sbsi_caches/flow_shape_augmentation_20260918_v1/matched_model_probe_smoke.json" \
  --device cpu --threads 4 --draws 64 --batch-size 256 --seed 7301 --operational-small-scene
