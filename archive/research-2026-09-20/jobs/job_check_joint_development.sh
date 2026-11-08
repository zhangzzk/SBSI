#!/usr/bin/env bash
# One-shot snapshot of current development jobs; no automatic model promotion.
#SBATCH --job-name=sbsi_joint_development_check
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/blendemu_runs/joint_fullscene_development_20260918_v1/logs/check_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/blendemu_runs/joint_fullscene_development_20260918_v1/logs/check_%j.err
set -euo pipefail
array_job=${1:?full-scene array job required}
validation_jobs=${2:-}
probe_job=${3:-}
all_jobs="$array_job,16563497,16563498${validation_jobs:+,$validation_jobs}${probe_job:+,$probe_job}"
date --iso-8601=seconds
squeue -j "$all_jobs" -o '%.20i %.28j %.10T %.10M %R' || true
sacct -j "$all_jobs" --format=JobID,State,Elapsed,ExitCode -n -X
python - "$array_job" "$validation_jobs" "$probe_job" <<'PY'
from pathlib import Path
import json, sys
base = Path('/project/ls-gruen/users/zekang.zhang')
root = base/'blendemu_runs/joint_fullscene_development_20260918_v1'
def tail(path):
    if path.exists():
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size-1800))
            print(str(path), stream.read().decode(errors='replace'))
for task in range(4):
    for suffix in ('out', 'err'):
        tail(root/f'logs/blendemu_joint_fullscene_{sys.argv[1]}_{task}.{suffix}')
    directory = root/f'case{20010+task}'
    for name in ('manifest.json', 'render_validation.json', 'measurement_manifest.json'):
        path = directory/name
        if path.exists():
            result = json.loads(path.read_text())
            print('FRESH_STAGE', task, name, result.get('status'), result.get('generated_rows'))
    logs = sorted(directory.glob('render_*.log'), key=lambda p: p.stat().st_mtime)
    if logs:
        tail(logs[-1])
training = base/'sbsi_caches/flow_shape_augmentation_20260918_v1'
for arm, job in [('control', 16563497), ('augmented', 16563498)]:
    for suffix in ('out', 'err'):
        tail(training/f'logs/sbsi_shape_augment_{job}.{suffix}')
    path = training/arm/'completed.json'
    if path.exists():
        print('TRAINING_COMPLETED', arm, path)
for job in filter(None, sys.argv[2].split(',')):
    for prefix in ('sbsi_classifier_selected', 'sbsi_classifier_influence', 'sbsi_continuous_crowding',
                   'sbsi_smooth_classifier', 'sbsi_smooth_smoke', 'sbsi_smooth_freeze', 'sbsi_smooth_joint',
                   'sbsi_smooth_density', 'sbsi_smooth_density_tests', 'sbsi_smooth_influence',
                   'sbsi_smooth_latent_repeat', 'sbsi_smooth_replica', 'sbsi_smooth_replica_density',
                   'sbsi_smooth_replica_compare'):
        for path in sorted((training/'logs').glob(f'{prefix}_{job}*')):
            tail(path)
    for prefix in ('sbsi_shape_joint', 'sbsi_shape_compare', 'sbsi_flow_per_leg', 'sbsi_shape_density', 'sbsi_per_leg_cv', 'sbsi_guard_normalization', 'sbsi_initial_main_compare', 'sbsi_per_leg_refine', 'sbsi_per_leg_joint', 'sbsi_joint_gradient', 'sbsi_joint_prob_refine', 'sbsi_joint_prob_validate', 'sbsi_matched_model_full', 'sbsi_own_guard_audit', 'sbsi_classifier_transfer', 'sbsi_coherent_classifier', 'sbsi_usability_coupling', 'sbsi_scene_classifier_prep', 'sbsi_scene_classifier_train', 'sbsi_scene_classifier_valprep', 'sbsi_scene_classifier_suite', 'sbsi_scene_classifier_joint_smoke', 'sbsi_scene_classifier_joint_full', 'sbsi_scene_classifier_calibration', 'sbsi_signed_classifier', 'sbsi_classifier_train_response', 'sbsi_geometry_coordinates', 'sbsi_transported_geometry_prep', 'sbsi_transported_classifier', 'sbsi_transported_joint_smoke', 'sbsi_transported_classifier_audit', 'sbsi_transported_joint_freeze', 'sbsi_transported_joint_full', 'sbsi_transported_receipt_tests', 'sbsi_transported_suite'):
        for path in sorted((training/'logs').glob(f'{prefix}_{job}*')):
            tail(path)
fresh = base/'sbsi_caches/fresh_fullscene_joint_20260918_v1'
for job in filter(None, sys.argv[2].split(',')):
    for prefix in ('sbsi_fresh_parent', 'sbsi_fresh_joint', 'sbsi_fresh_compare'):
        for path in sorted((fresh/'logs').glob(f'{prefix}_{job}*')):
            tail(path)
for scope in ('smoke', 'full'):
    path = fresh/scope/'preparation.json'
    if path.exists():
        print('FRESH_PREPARATION', scope, json.loads(path.read_text())['status'])
    for path in sorted((fresh/scope/'results').glob('*.json')):
        print('FRESH_JOINT_RESULT', scope, path.name)
for arm in ('control', 'augmented'):
    for path in sorted([*(training/arm).glob('joint_*.json'), *(training/arm).glob('transported_density_*.json')]):
        print('JOINT_RESULT', arm, path.name)
if sys.argv[3]:
    probe = base/'blendemu_runs/random_sign_probe_fullscene_20260918_v1'
    for suffix in ('out', 'err'):
        tail(probe/f'logs/blendemu_random_sign_full_{sys.argv[3]}.{suffix}')
    for name in ('manifest.json', 'render_validation.json', 'measurement_manifest.json', 'response_diagnostic.json'):
        path = probe/'images'/name
        if path.exists():
            print('FULL_PROBE_STAGE', name, json.loads(path.read_text()).get('status'))
PY

python - <<'PYCODE'
from pathlib import Path
root = Path('/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1')
for path in root.glob('model_per_leg_*.json'):
    print('MODEL_PER_LEG_RESULT', path)
for path in root.glob('guard_normalization_*.json'):
    print('GUARD_NORMALIZATION_RESULT', path)
for path in root.glob('per_leg_cv_*/result.json'):
    print('PER_LEG_GRADIENT_RESULT', path)
for path in root.glob('per_leg_refinement_*/completed.json'):
    print('PER_LEG_REFINEMENT_RESULT', path)
for directory in ('per_leg_refinement_full', 'joint_probability_refinement_full'):
    for path in (root/directory).glob('validation_*_status.json'):
        print('CANDIDATE_VALIDATION_STATUS', path, path.read_text())
for path in root.glob('joint_gradient_*/result.json'):
    print('JOINT_GRADIENT_RESULT', path)
for path in root.glob('joint_probability_refinement_*/completed.json'):
    print('JOINT_PROBABILITY_REFINEMENT_RESULT', path)
for path in root.glob('matched_model_probe_full_seed*.json'):
    print('MATCHED_MODEL_PROBE_RESULT', path)
scene_classifier = root.parent/'scene_classifier_geometry_20260918_v1'
for pattern in ('classifier_selected_density_*.json', 'continuous_crowding*.json',
                'smooth_*.json', 'smooth_classifier_*/completed.json', 'smooth_classifier_*/history.json',
                'replica_results/*.json'):
    for path in scene_classifier.glob(pattern):
        print('CLASSIFIER_DENSITY_AUDIT', path)
for path in scene_classifier.glob('full_*/completed.json'):
    print('SCENE_CLASSIFIER_TRAINING_RESULT', path)
for pattern in ('signed_response*.json', 'training_response*.json', 'coordinate_*.json', 'transported_*.json', 'transported_classifier_*/completed.json', 'transported_classifier_*/history.json', 'transported_prepared_*/manifest.json'):
    for path in scene_classifier.glob(pattern):
        print('CLASSIFIER_RESPONSE_AUDIT', path)
for path in scene_classifier.glob('calibration_*.json'):
    print('SCENE_CLASSIFIER_CALIBRATION_RESULT', path)
for path in scene_classifier.glob('joint_*.json'):
    print('SCENE_CLASSIFIER_JOINT_RESULT', path)
for path in root.glob('usability_blend_coupling*.json'):
    print('USABILITY_BLEND_COUPLING_RESULT', path)
for path in root.glob('coherent_classifier_*.json'):
    print('COHERENT_CLASSIFIER_RESULT', path)
for path in root.glob('classifier_transfer_*.json'):
    print('CLASSIFIER_TRANSFER_RESULT', path)
for path in root.glob('own_normalized_guard_*/result.json'):
    print('OWN_NORMALIZED_GUARD_RESULT', path)
PYCODE
