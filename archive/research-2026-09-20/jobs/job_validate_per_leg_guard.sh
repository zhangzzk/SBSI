#!/usr/bin/env bash
#SBATCH --job-name=sbsi_per_leg_guard
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
base=/project/ls-gruen/users/zekang.zhang/sbsi_caches
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SBSI_MULTIAXIS_EVALUATOR=$base/joint_multiaxis_evaluator_20260918_v1/evaluate_constgold_truth_parent.py
cd /home/z/Zekang.Zhang/SBSI
"$python" -m pytest tests -q
"$python" -u -m scripts.per_leg_guard_response \
  --domain-root "$base/harmonized_secondary_20260917_v1/domain" --split validation \
  --output "$base/flow_shape_augmentation_20260918_v1/per_leg_guard_validation_targets.json"
"$python" - "$base/flow_shape_augmentation_20260918_v1" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
targets = json.loads((root/'per_leg_guard_validation_targets.json').read_text())
old = json.loads((root/'control/history.json').read_text())['history'][0]['validation_guard']['target']
np.testing.assert_allclose(targets['paired']['target'], old, rtol=0, atol=2e-12)
label = json.loads((root/'validation_usability_response.json').read_text())
indices = [0, next(i for i,c in enumerate(targets['per_leg']['cuts']) if c['name'] == 'radius_gt_3_magnitude_lt_25.8')]
np.testing.assert_allclose(np.array(targets['per_leg']['target'])[indices].reshape(2,2,2),
                         label['responses']['soft/per_leg']['estimate'], rtol=0, atol=2e-12)
print('PER_LEG_GUARD_FULL_LABEL_REFERENCE_CHECK_PASSED', flush=True)
PY
