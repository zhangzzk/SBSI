#!/usr/bin/env bash

#SBATCH --job-name=sbsi_detshape
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_classifier_rendered_shape_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_classifier_rendered_shape_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
simulation_root=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_classifier_rendered_shape_v1
output=$root/models
deployed=$repo/models/blendemu/classification_model_lsst_r_extnbr_ho.json

export PYTHONPATH=$repo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p "$root/logs"
for path in "$simulation_root" "$deployed"; do
  [[ -e "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ ! -e "$output/report.json" ]] || { echo "refusing to overwrite $output/report.json" >&2; exit 2; }

"$python" -u "$repo/scripts/train_detection_classifier_ladder.py" \
  --simulation-root "$simulation_root" --output-dir "$output" \
  --train-cases 0-29 --validation-cases 30-39 \
  --zero-shear-label 0.0 --forward-shear-label 0.05 \
  --radius-arcsec 3 --impact-exponents 1 2 4 \
  --max-primaries-per-case 50000 --sampling-seed 20260824 \
  --xgb-seed 321 --device cuda --nthread "$SLURM_CPUS_PER_TASK" \
  --num-boost-round 800 --early-stopping-rounds 30 \
  --deployed-model "$deployed"
