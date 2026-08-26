#!/usr/bin/env bash

#SBATCH --job-name=sbsi_detcgold
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_classifier_constgold_c40_49_v2/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_classifier_constgold_c40_49_v2/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
simulation_root=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_classifier_constgold_c40_49_v2
output=$root/report.json
ordinary=/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_classifier_transition_pair_v1/models/ordinary_concatenation.pt
transition=/project/ls-gruen/users/zekang.zhang/sbsi_caches/detection_classifier_transition_lambda1_v1/models/transition_aware.pt

export PYTHONPATH=$repo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p "$root/logs"
[[ -d "$simulation_root" ]] || { echo "missing input: $simulation_root" >&2; exit 2; }
[[ -f "$ordinary" ]] || { echo "missing input: $ordinary" >&2; exit 2; }
[[ -f "$transition" ]] || { echo "missing input: $transition" >&2; exit 2; }
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

"$python" -u "$repo/scripts/evaluate_detection_classifier_constgold.py" \
  --simulation-root "$simulation_root" --output "$output" \
  --cases 40-49 --shear-magnitude 0.02 \
  --model "ordinary_concatenation=$ordinary" \
  --model "transition_aware=$transition" \
  --radius-arcsec 3 --impact-exponent 1 --max-primaries-per-case 0 \
  --sampling-seed 20260824 --prediction-batch-size 65536 --device cuda
