#!/usr/bin/env bash
#SBATCH --job-name=sbsi_initial_response
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/flow_shape_augmentation_20260918_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
"$python" -m pytest tests/test_truth_parent_projection.py tests/test_truth_parent_comparison.py -q
"$python" -m scripts.project_truth_parent_audit \
  --source "$cache/harmonized_secondary_20260917_v1/candidate_joint.json" \
  --output "$root/initial_main_truth_parent.json"
"$python" -m scripts.compare_truth_parent_response \
  --baseline "$root/initial_main_truth_parent.json" --candidate "$root/initial_main_truth_parent.json" \
  --output "$root/initial_main_projection_selfcheck.json"
"$python" - "$root/initial_main_projection_selfcheck.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
for value in result['comparisons'].values():
    assert value['delta_m_percentage_points'] == 0
    assert value['paired_delta_se_pp'] == 0
    assert value['paired_delta_ci95_pp'] == [0, 0]
print('TRUTH_PARENT_PROJECTION_SELF_COMPARISON_EXACT_ZERO', flush=True)
PY
