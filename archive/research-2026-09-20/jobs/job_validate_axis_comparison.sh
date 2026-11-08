#!/usr/bin/env bash
#SBATCH --job-name=sbsi_axis_comparison_test
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fresh_fullscene_joint_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fresh_fullscene_joint_20260918_v1/logs/%x_%j.err
set -euo pipefail
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
base=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SBSI_MULTIAXIS_EVALUATOR=$base/joint_multiaxis_evaluator_20260918_v1/evaluate_constgold_truth_parent.py
cd /home/z/Zekang.Zhang/SBSI
"$python" -m pytest tests -q
"$python" -u -m scripts.compare_truth_parent_response \
  --baseline "$base/harmonized_secondary_20260917_v1/independent_baseline_full.json" \
  --candidate "$base/harmonized_secondary_20260917_v1/independent_candidate_full.json" \
  --output "$base/fresh_fullscene_joint_20260918_v1/comparison_validation/g1_replay.json"
"$python" - "$base" <<'PY'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
old = json.loads((base/'harmonized_secondary_20260917_v1/independent_joint_comparison.json').read_text())
new = json.loads((base/'fresh_fullscene_joint_20260918_v1/comparison_validation/g1_replay.json').read_text())
if old['comparisons'] != new['comparisons']:
    raise RuntimeError('g1 comparison changed after explicit-axis extension')
print('FULL_40_CASE_G1_COMPARISON_EXACT_REPLAY_PASSED', flush=True)
PY
