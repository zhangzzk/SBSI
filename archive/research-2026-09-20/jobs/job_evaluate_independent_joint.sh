#!/usr/bin/env bash
# Compare frozen flows on distinct existing g1 scenes; no checkpoint selection.
# Use the validated frozen evaluator copy while the original source is in use.
# Chain each arm behind one occupied GPU slot; at most two SBSI GPUs total.
#SBATCH --job-name=sbsi_independent_joint
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.err
set -euo pipefail
data=/project/ls-gruen/users/zekang.zhang
cache=$data/sbsi_caches
root=$cache/harmonized_secondary_20260917_v1
python=$data/envs/py31/bin/python
evaluator=${SBSI_EVALUATOR_PATH:-/home/z/Zekang.Zhang/SBSI/scripts/evaluate_constgold_truth_parent.py}
arm=${1:?baseline or candidate}
scope=${2:?smoke or full}
case $arm in
  baseline) flow=$cache/full_domain_true_mag26_guard_v1/flow/guard/selected.pt ;;
  candidate) flow=$root/refinement/selected.pt ;;
  *) exit 2 ;;
esac
case $scope in
  smoke) cases=(200 201) ;;
  full) mapfile -t cases < <(seq 200 239) ;;
  *) exit 2 ;;
esac
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
"$python" - "$root" "$flow" "$arm" "$scope" "$evaluator" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root, flow = map(Path, sys.argv[1:3])
arm, scope = sys.argv[3:5]
evaluator = Path(sys.argv[5])
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
staging = json.loads((root/'evaluator_staging/manifest.json').read_text())
if digest(evaluator) != staging['staged_sha256']:
    raise RuntimeError('evaluator does not match the validated staged source')
if digest(Path('scripts/truth_parent_cohorts.py')) != staging['helper_sha256']:
    raise RuntimeError('cohort helper changed after staging')
if digest(Path('scripts/joint_predictive_diagnostics.py')) != staging['predictive_helper_sha256']:
    raise RuntimeError('predictive diagnostics changed after staging')
selected = json.loads((root/'refinement/completed.json').read_text())
expected = (selected['selected_sha256'] if arm == 'candidate' else
            '6755f4e00fce016c3fe94c5db7c41f70ffc3f3b33b6e2a6db8124720c25d0859')
if digest(flow) != expected or selected['full_density_gate_passed'] is not True:
    raise RuntimeError('frozen flow or independent training density gate mismatch')
prepared = json.loads((root/'independent_pair_preparation.json').read_text())
if prepared['status'] != 'completed' or prepared['cases'] != list(range(200, 240)):
    raise RuntimeError('independent pair preparation incomplete')
if scope == 'full':
    smoke = json.loads((root/f'independent_{arm}_smoke.json').read_text())
    if (smoke['cases'] != [200, 201] or smoke['cohorts'] != ['truth_parent']
            or smoke['h'] != 0.05 or smoke['seed'] != 7301 or smoke['draws'] != 64
            or smoke['inputs']['flow_sha256'] != expected
            or smoke['inputs']['script_sha256'] != staging['staged_sha256']
            or smoke['inputs']['cohort_helper_sha256'] != staging['helper_sha256']
            or smoke['inputs']['predictive_helper_sha256'] != staging['predictive_helper_sha256']
            or smoke.get('predictive_diagnostics') is None):
        raise RuntimeError('matching independent smoke result required')
PY
options=()
for seed in 20260913 20260914 20260915; do
  options+=(--classifier "$cache/fixed_g0_m258_r060_v1/classifier/seed${seed}/selected.pt")
done
for first in $(seq 200 10 230); do
  options+=(--rblend "$root/independent_pair_lookups/cases${first}_$((first+9)).feather")
done
exec "$python" -u -c 'import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name="__main__")' "$evaluator" \
  --parent-root "$root/independent_parent" \
  --constgold-root "$data/lsst_sims_fs2_25876_constant_g005_c140-239" \
  --flow "$flow" --response-candidate "$data/blendemu_runs/disk_response_moment_20260918_v1/model" \
  --response-composition mobius --response-device cuda --response-threads 8 \
  --pair-root "$root/independent_pairs" --cohorts truth_parent --predictive-diagnostics \
  --h 0.05 --seed 7301 --draws 64 --batch-size 1024 \
  "${options[@]}" --cases "${cases[@]}" --output "$root/independent_${arm}_${scope}.json"
