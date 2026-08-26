#!/usr/bin/env bash

#SBATCH --job-name=sbsi_detpair
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_classifier_transition_pair_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_classifier_transition_pair_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
simulation_root=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_classifier_transition_pair_v1
output=$root/models

export PYTHONPATH=$repo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p "$root/logs"
[[ -d "$simulation_root" ]] || { echo "missing input: $simulation_root" >&2; exit 2; }
[[ ! -e "$output/report.json" ]] || { echo "refusing to overwrite $output/report.json" >&2; exit 2; }

"$python" -u "$repo/scripts/train_detection_classifier_transition_pair.py" \
  --simulation-root "$simulation_root" --output-dir "$output" \
  --train-cases 0-24 --tune-cases 25-29 --test-cases 30-39 \
  --zero-shear-label 0.0 --forward-shear-label 0.05 \
  --radius-arcsec 3 --impact-exponent 1 \
  --max-primaries-per-case 50000 --sampling-seed 20260824 \
  --model-seed 20260825 --hidden-dim 128 --n-layers 3 \
  --batch-size 16384 --prediction-batch-size 65536 \
  --epochs 40 --patience 6 --learning-rate 0.001 --weight-decay 0.0001 \
  --transition-weight 0.25 --num-workers 4 --device cuda
