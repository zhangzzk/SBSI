#!/usr/bin/env bash
# Frozen single-flow comparison; checkpoint selection used training validation only.
#SBATCH --job-name=sbsi_shape_joint
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
arm=${1:?control or augmented}
endpoint=${2:?main or independent}
scope=${3:?smoke or full}
case $arm in control|augmented) ;; *) exit 2 ;; esac
case $endpoint in main|independent) ;; *) exit 2 ;; esac
case $scope in smoke|full) ;; *) exit 2 ;; esac
data=/project/ls-gruen/users/zekang.zhang
cache=$data/sbsi_caches
root=$cache/flow_shape_augmentation_20260918_v1
harmonized=$cache/harmonized_secondary_20260917_v1
python=$data/envs/py31/bin/python
flow=$root/$arm/selected.pt
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
"$python" - "$root" "$harmonized" "$arm" "$endpoint" "$scope" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root, harmonized = map(Path, sys.argv[1:3])
arm, endpoint, scope = sys.argv[3:]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
record = json.loads((root/arm/'completed.json').read_text())
expected = {'control': '1829b31cfeca33ab368c153d2d7a0b11e369484bcdbfdede385df40eed2de874',
            'augmented': 'a88af411e3d7875d51e42c63c3741c10dde3a92f081334427beb335a9633f4e3'}[arm]
if (record['arm'] != arm or record['smoke'] is not False
        or record['eligible_for_joint_validation'] is not True
        or record['selected_sha256'] != expected
        or digest(root/arm/'selected.pt') != expected
        or digest(root/arm/'protocol.json') != record['protocol_sha256']
        or record['selected_full_validation_nll'] > record['baseline_full_validation_nll']
        or record['selected_guard_loss'] > record['baseline_guard_loss']):
    raise RuntimeError('training-only selection or checkpoint identity mismatch')
staging = json.loads((harmonized/'evaluator_staging/manifest.json').read_text())
for filename, key in [('evaluate_constgold_truth_parent.py', 'staged_sha256'),
                      ('truth_parent_cohorts.py', 'helper_sha256'),
                      ('joint_predictive_diagnostics.py', 'predictive_helper_sha256')]:
    if digest(Path('scripts')/filename) != staging[key]:
        raise RuntimeError(f'frozen evaluator source changed: {filename}')
if scope == 'full':
    smoke = json.loads((root/arm/f'joint_{endpoint}_smoke.json').read_text())
    start, h = (40, 0.02) if endpoint == 'main' else (200, 0.05)
    if (smoke['cases'] != [start, start+1] or smoke['h'] != h
            or smoke['cohorts'] != ['truth_parent'] or smoke['seed'] != 7301
            or smoke['draws'] != 64 or smoke.get('predictive_diagnostics') is None
            or smoke['inputs']['flow_sha256'] != expected
            or smoke['inputs']['script_sha256'] != staging['staged_sha256']
            or smoke['inputs']['cohort_helper_sha256'] != staging['helper_sha256']
            or smoke['inputs']['predictive_helper_sha256'] != staging['predictive_helper_sha256']):
        raise RuntimeError('matching successful smoke result required')
print('FROZEN_SINGLE_FLOW_VALIDATION', arm, endpoint, scope, expected, flush=True)
PY
options=()
for seed in 20260913 20260914 20260915; do
  options+=(--classifier "$cache/fixed_g0_m258_r060_v1/classifier/seed${seed}/selected.pt")
done
if [[ $endpoint == main ]]; then
  first=40; last=119; h=0.02
  parent=$cache/constgold_truth_parent_20260917_v1/parent_v2
  catalog=$data/lsst_sims_fs2_25876_constant
  pairs=$cache/output_conditioned_response_20260917_v1/pairs
  for block in $(seq 40 10 110); do
    options+=(--rblend "$cache/constgold_truth_parent_20260917_v1/rblend/cases${block}_$((block+9)).feather")
  done
else
  first=200; last=239; h=0.05
  parent=$harmonized/independent_parent
  catalog=$data/lsst_sims_fs2_25876_constant_g005_c140-239
  pairs=$harmonized/independent_pairs
  for block in $(seq 200 10 230); do
    options+=(--rblend "$harmonized/independent_pair_lookups/cases${block}_$((block+9)).feather")
  done
fi
if [[ $scope == smoke ]]; then cases=("$first" "$((first+1))"); else mapfile -t cases < <(seq "$first" "$last"); fi
exec "$python" -u -m scripts.evaluate_constgold_truth_parent \
  --parent-root "$parent" --constgold-root "$catalog" --flow "$flow" \
  --response-candidate "$data/blendemu_runs/disk_response_moment_20260918_v1/model" \
  --response-composition mobius --response-device cuda --response-threads 8 \
  --pair-root "$pairs" --cohorts truth_parent --predictive-diagnostics \
  --h "$h" --seed 7301 --draws 64 --batch-size 1024 \
  "${options[@]}" --cases "${cases[@]}" --output "$root/$arm/joint_${endpoint}_${scope}.json"
