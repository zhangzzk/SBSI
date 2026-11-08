#!/usr/bin/env bash
# CPU operational smoke by default. Full array waits for all existing GPU chains.
#SBATCH --job-name=sbsi_fresh_joint
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fresh_fullscene_joint_20260918_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fresh_fullscene_joint_20260918_v1/logs/%x_%A_%a.err
set -euo pipefail
mode=${1:?smoke or full}
task=${SLURM_ARRAY_TASK_ID:-0}
if (( task < 0 || task > 5 )); then exit 2; fi
arms=(initial control augmented)
arm=${arms[$((task/2))]}
axis=$((task%2+1))
data=/project/ls-gruen/users/zekang.zhang
cache=$data/sbsi_caches
root=$cache/fresh_fullscene_joint_20260918_v1
staged=$cache/joint_multiaxis_evaluator_20260918_v1
augmentation=$cache/flow_shape_augmentation_20260918_v1
python=$data/envs/py31/bin/python
flow=$augmentation/$arm/selected.pt
if [[ $arm == initial ]]; then flow=$cache/harmonized_secondary_20260917_v1/refinement/selected.pt; fi
case $mode in
  smoke) prepared=$root/smoke; cases=(20000); device=cpu; options=(--operational-single-case) ;;
  full) prepared=$root/full; cases=(20010 20011 20012 20013); device=cuda; options=(--predictive-diagnostics) ;;
  *) exit 2 ;;
esac
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
"$python" - "$root" "$staged" "$augmentation" "$flow" "$arm" "$mode" "$prepared" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, staged, augmentation, flow = map(Path, sys.argv[1:5])
arm, mode = sys.argv[5:7]
prepared = Path(sys.argv[7])
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((staged/'manifest.json').read_text())
if (manifest['status'] != 'validated'
        or digest(staged/'evaluate_constgold_truth_parent.py') != manifest['staged_sha256']
        or digest(Path('scripts/joint_response_axes.py')) != manifest['axis_helper_sha256']):
    raise RuntimeError('validated axis evaluator changed')
protocol = json.loads((augmentation/'joint_validation_protocol.json').read_text())
expected = ('fade8aebfb4825344b8e9e36dfc07fce5357649340995670f1bb284df7068bd4'
            if arm == 'initial' else protocol['arms'][arm]['selected_sha256'])
if digest(flow) != expected:
    raise RuntimeError('frozen flow identity changed')
for path in ('scripts/joint_predictive_diagnostics.py', 'scripts/truth_parent_cohorts.py'):
    if digest(Path(path)) != protocol['source_sha256'][path]:
        raise RuntimeError('frozen cohort or density helper changed')
receipt = json.loads((prepared/'preparation.json').read_text())
if (receipt['status'] != 'completed' or receipt['operational_small_scene'] != (mode == 'smoke')
        or receipt['parent_manifest_sha256'] != digest(prepared/'parent/manifest.json')):
    raise RuntimeError('prepared development population changed')
if mode == 'full':
    for axis in (1, 2):
        smoke = json.loads((root/f'smoke/results/initial_g{axis}_h0.02.json').read_text())
        if (smoke['cases'] != [20000] or smoke['h'] != .02 or smoke['shear_axis'] != axis
                or smoke['operational_single_case'] is not True
                or smoke['inputs']['script_sha256'] != manifest['staged_sha256']):
            raise RuntimeError('matching two-axis operational smoke required')
    for group in ('g1_h0.02', 'g2_h0.02'):
        if receipt['group_cases'][group] != [20010, 20011, 20012, 20013]:
            raise RuntimeError('full four-case shear group required')
    if receipt['group_cases']['g1_h0.04'] != [20010] or receipt['group_cases']['g2_h0.04'] != [20010]:
        raise RuntimeError('explicit one-scene amplitude pilot required')
print('FRESH_JOINT_VALIDATED', arm, mode, flush=True)
PY
mkdir -p "$prepared/results"
classifiers=()
for seed in 20260913 20260914 20260915; do classifiers+=(--classifier "$cache/fixed_g0_m258_r060_v1/classifier/seed${seed}/selected.pt"); done
run_evaluator() {
  local step=$1
  shift
  "$python" -u -c 'import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name="__main__")' \
    "$staged/evaluate_constgold_truth_parent.py" --parent-root "$prepared/parent" \
    --constgold-root "$prepared/catalogues/g${axis}_h${step}" --flow "$flow" \
    --response-candidate "$data/blendemu_runs/disk_response_moment_20260918_v1/model" \
    --response-composition mobius --response-device "$device" --response-threads 4 --device "$device" \
    --pair-root "$prepared/pairs" --rblend "$prepared/pair_lookup.feather" --cohorts truth_parent \
    --h "$step" --axis "$axis" --seed 7301 --draws 64 --batch-size 1024 \
    "${classifiers[@]}" --output "$prepared/results/${arm}_g${axis}_h${step}.json" "$@"
}
run_evaluator 0.02 --cases "${cases[@]}" "${options[@]}"
if [[ $mode == full ]]; then run_evaluator 0.04 --cases 20010 --operational-single-case; fi
