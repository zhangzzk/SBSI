#!/usr/bin/env bash
#SBATCH --job-name=sbsi_per_leg_cv
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=38G
#SBATCH --time=00:15:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
mode=${1:?smoke or full}
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/flow_shape_augmentation_20260918_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
case "$mode" in
  smoke) options=(--smoke --device cpu) ;;
  full)
    options=(--device cuda)
    "$python" - "$root" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
smoke = root/'per_leg_cv_smoke'
protocol = json.loads((smoke/'protocol.json').read_text())
result = json.loads((smoke/'result.json').read_text())
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
if (result['smoke'] is not True or result['optimizer_steps'] != 0
        or result['protocol_sha256'] != digest(smoke/'protocol.json')):
    raise RuntimeError('matching completed gradient smoke required')
identities = {'script_sha256': 'scripts/audit_per_leg_guard_gradient.py',
    'guard_adapter_sha256': 'scripts/per_leg_guard_response.py',
    'gradient_helper_sha256': 'sbsi/flow_guard_response.py',
    'label_helper_sha256': 'scripts/audit_harmonized_usability_response.py',
    'target_bank_sha256': root/'per_leg_guard_train_targets.json',
    'flow_sha256': root.parent/'harmonized_secondary_20260917_v1/refinement/selected.pt',
    'domain_manifest_sha256': root.parent/'harmonized_secondary_20260917_v1/domain/manifest.json'}
for key, path in identities.items():
    if protocol[key] != digest(path):
        raise RuntimeError(f'changed operational smoke identity: {key}')
PY
    ;;
  *) exit 2 ;;
esac
exec "$python" -u -m scripts.audit_per_leg_guard_gradient \
  --domain-root "$cache/harmonized_secondary_20260917_v1/domain" \
  --flow "$cache/harmonized_secondary_20260917_v1/refinement/selected.pt" \
  --target-bank "$root/per_leg_guard_train_targets.json" \
  --output-root "$root/per_leg_cv_${mode}" "${options[@]}"
