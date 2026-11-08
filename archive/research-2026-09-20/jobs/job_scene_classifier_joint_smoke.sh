#!/usr/bin/env bash
#SBATCH --job-name=sbsi_scene_classifier_joint_smoke
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
base=/project/ls-gruen/users/zekang.zhang
cache=$base/sbsi_caches
root=$cache/scene_classifier_geometry_20260918_v1
prepared=$cache/fresh_fullscene_joint_20260918_v1/smoke
protocol=${1:?frozen smoke execution protocol}
"$python" - "$protocol" "$root" <<'CHECK'
from pathlib import Path
import hashlib,json,sys
p=json.loads(Path(sys.argv[1]).read_text())
for path,expected in p['files'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Joint classifier smoke input changed: '+path)
for arm in ['control','geometry']:
    root=Path(sys.argv[2])/('full_'+arm)
    r=json.loads((root/'completed.json').read_text())
    if r['smoke'] or r['response_read'] or hashlib.sha256((root/'selected.pt').read_bytes()).hexdigest()!=r['selected_sha256']:
        raise RuntimeError('Successful frozen classifier training required')
print('SCENE_CLASSIFIER_JOINT_SMOKE_PROTOCOL_VERIFIED',flush=True)
CHECK
"$python" -m pytest tests/test_scene_classifier_joint.py tests/test_scene_classifier_validation.py -q
"$python" -u -m scripts.prepare_scene_classifier --anchor-root "$prepared/parent" \
 --pair-root "$prepared/pairs" --output-root "$root/validation_smoke" --cases 20000
args=(--parent-root "$prepared/parent" --constgold-root "$prepared/catalogues/g1_h0.02" \
 --pair-root "$prepared/pairs" --rblend "$prepared/pair_lookup.feather" \
 --flow "$cache/flow_shape_augmentation_20260918_v1/joint_probability_refinement_full/selected.pt" \
 --response-candidate "$base/blendemu_runs/disk_response_moment_20260918_v1/model" \
 --response-composition mobius --response-device cpu --response-threads 4 \
 --cohorts truth_parent --cases 20000 --h 0.02 --axis 1 --device cpu \
 --draws 64 --batch-size 256 --seed 7301 --operational-single-case)
for seed in 20260913 20260914 20260915; do
 args+=(--classifier "$cache/fixed_g0_m258_r060_v1/classifier/seed$seed/selected.pt")
done
"$python" -u -c 'import runpy,sys; runpy.run_path(sys.argv.pop(1),run_name="__main__")' \
 "$cache/joint_multiaxis_evaluator_20260918_v1/evaluate_constgold_truth_parent.py" \
 "${args[@]}" --output "$root/joint_smoke_reference.json"
"$python" -u -m scripts.evaluate_scene_classifier_joint "${args[@]}" \
 --classifier-control "$root/full_control/selected.pt" --classifier-geometry "$root/full_geometry/selected.pt" \
 --geometry-root "$root/validation_smoke" --output "$root/joint_smoke_arms.json"
"$python" - "$root" <<'CHECK'
from pathlib import Path
import json,sys,hashlib
import numpy as np
root=Path(sys.argv[1]);a=json.loads((root/'joint_smoke_reference.json').read_text());b=json.loads((root/'joint_smoke_arms.json').read_text())
maximum=0.
for key, sources in a['per_case_sufficient'].items():
    for source,legs in sources.items():
        for leg,fields in legs.items():
            for field,value in fields.items():
                left=np.asarray(value);right=np.asarray(b['per_case_sufficient'][key][source][leg][field])
                np.testing.assert_allclose(left,right,rtol=1e-12,atol=1e-10)
                maximum=max(maximum,float(np.max(np.abs(left-right))))
for key,summary in a['summary'].items():
    for field in ['measured_response','model_response','model_flow_response','model_blend_response']:
        np.testing.assert_allclose(summary[field],b['summary'][key][field],rtol=1e-12,atol=1e-12)
receipt={'scope':'operational small-prior replay, not calibration','reference_branches_replayed':True,
         'maximum_sufficient_absolute_difference':maximum,'classifier_arms':b['inputs']['classifier_arms'],
         'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'joint_smoke_reference.json',root/'joint_smoke_arms.json']}}
(root/'joint_smoke_completed.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('SCENE_CLASSIFIER_REFERENCE_REPLAY_PASSED',maximum,flush=True)
CHECK
