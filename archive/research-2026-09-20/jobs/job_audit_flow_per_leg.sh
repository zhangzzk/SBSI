#!/usr/bin/env bash
# CPU smoke by default; full audit uses one GPU after the existing control chain.
#SBATCH --job-name=sbsi_flow_per_leg
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
mode=${1:?smoke or full}
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1
harmonized=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
case $mode in
  smoke) options=(--smoke --device cpu) ;;
  full) options=(--device cuda) ;;
  *) exit 2 ;;
esac
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
"$python" - "$root" "$harmonized" "$mode" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, harmonized = map(Path, sys.argv[1:3])
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
protocol = json.loads((root/'joint_validation_protocol.json').read_text())
expected = {'initial': 'fade8aebfb4825344b8e9e36dfc07fce5357649340995670f1bb284df7068bd4'}
expected.update({arm: protocol['arms'][arm]['selected_sha256'] for arm in ('control', 'augmented')})
for arm, sha in expected.items():
    path = harmonized/'refinement/selected.pt' if arm == 'initial' else root/arm/'selected.pt'
    if digest(path) != sha:
        raise RuntimeError('changed frozen checkpoint')
if sys.argv[3] == 'full':
    smoke = json.loads((root/'model_per_leg_smoke.json').read_text())
    if (smoke['smoke'] is not True or smoke['flow_sha256'] != expected
            or smoke['script_sha256'] != digest(Path('scripts/audit_flow_per_leg_response.py'))
            or smoke['label_helper_sha256'] != digest(Path('scripts/audit_harmonized_usability_response.py'))
            or smoke['domain_manifest_sha256'] != digest(harmonized/'domain/manifest.json')):
        raise RuntimeError('matching successful model-side smoke required')
PY
"$python" -m pytest tests/test_flow_per_leg_response.py -q
exec "$python" -u -m scripts.audit_flow_per_leg_response \
  --domain-root "$harmonized/domain" \
  --flow "initial=$harmonized/refinement/selected.pt" \
  --flow "control=$root/control/selected.pt" --flow "augmented=$root/augmented/selected.pt" \
  --output "$root/model_per_leg_${mode}.json" "${options[@]}"
