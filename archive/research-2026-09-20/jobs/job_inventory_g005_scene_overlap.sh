#!/usr/bin/env bash
#SBATCH --job-name=sbsi_g005_scene_overlap
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u - <<'PY'
from pathlib import Path
import json
from scripts.audit_existing_g005_truth import generated_scene
from scripts.train_fixed_g0_flow import sha256, write_json

base = Path('/project/ls-gruen/users/zekang.zhang')
root = base/'sbsi_caches/harmonized_secondary_20260917_v1'
prior_path = root/'existing_g005_truth.json'
output = root/'existing_g005_scene_overlap.json'
if output.exists():
    raise FileExistsError(output)
prior = json.loads(prior_path.read_text())
if prior['script_sha256'] != sha256('scripts/audit_existing_g005_truth.py'):
    raise RuntimeError('fingerprint implementation changed since training-scene audit')
training = {value: int(case) for case, value in prior['training_scene_fingerprints'].items()}
reports, seen = {}, set()
for case in range(140, 240):
    paths = [base/'lsst_sims_fs2_25876_constant_g005_c140-239'/f'gals{case}_{h}.feather'
             for h in (.05, -.05)]
    fingerprints = [generated_scene(path) for path in paths]
    if fingerprints[0] != fingerprints[1] or fingerprints[0] in seen:
        raise RuntimeError('different antithetic scenes or duplicate independent case')
    seen.add(fingerprints[0])
    if str(case) in prior['case_reports'] and fingerprints[0] != prior['case_reports'][str(case)]['generated_scene_fingerprint']:
        raise RuntimeError('scene changed since completed truth audit')
    reports[str(case)] = {'scene_fingerprint': fingerprints[0],
                          'matching_training_case': training.get(fingerprints[0]),
                          'source_sha256': {str(path): sha256(path) for path in paths}}
    print(f'SCENE_OVERLAP case={case} training_case={training.get(fingerprints[0])}', flush=True)
write_json(output, {'prior_truth_audit_sha256': sha256(prior_path), 'reports': reports,
                    'verified_disjoint_cases': [int(c) for c, r in reports.items() if r['matching_training_case'] is None],
                    'overlap_cases': [int(c) for c, r in reports.items() if r['matching_training_case'] is not None],
                    'no_measured_response_values_opened': True,
                    'limitations': ['position/identity fingerprint only; separate rendered-truth and measurement provenance gates remain',
                                    'historically evaluated g1-only endpoint, not fresh multi-axis acceptance']})
print(f'SCENE_OVERLAP_COMPLETE {output}', flush=True)
PY
