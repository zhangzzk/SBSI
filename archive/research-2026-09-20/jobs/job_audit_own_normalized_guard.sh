#!/usr/bin/env bash
#SBATCH --job-name=sbsi_own_guard_audit
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
base=/project/ls-gruen/users/zekang.zhang/sbsi_caches
scope=${1:?smoke or full required}
args=()
case "$scope" in
  smoke) args+=(--smoke --device cpu) ;;
  full)
    args+=(--device cuda)
    protocol=${2:?frozen full-audit protocol required}
    "$python" - "$protocol" <<'PY'
from pathlib import Path
import hashlib,json,sys
p=json.loads(Path(sys.argv[1]).read_text())
for path,expected in p['files'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Own-normalized audit input changed: '+path)
smoke=json.loads(Path(p['smoke_result']).read_text())
if smoke['optimizer_steps']!=0 or smoke['smoke'] is not True:
    raise RuntimeError('Successful read-only smoke required')
print('OWN_GUARD_FULL_PROTOCOL_VERIFIED',flush=True)
PY
    ;;
  *) exit 2 ;;
esac
"$python" -m pytest tests/test_own_normalized_guard.py -q
exec "$python" -u -m scripts.audit_own_normalized_guard_gradient \
  --domain-root "$base/harmonized_secondary_20260917_v1/domain" \
  --flow "$base/harmonized_secondary_20260917_v1/refinement/selected.pt" \
  --target-bank "$base/flow_shape_augmentation_20260918_v1/per_leg_guard_train_targets.json" \
  --output-root "$base/flow_shape_augmentation_20260918_v1/own_normalized_guard_${scope}" "${args[@]}"
