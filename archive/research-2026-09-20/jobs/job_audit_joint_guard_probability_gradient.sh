#!/usr/bin/env bash
#SBATCH --job-name=sbsi_joint_gradient
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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
case $mode in
  smoke) options=(--smoke --device cpu); reference=$root/per_leg_cv_smoke ;;
  full) options=(--device cuda); reference=$root/per_leg_cv_full ;;
  *) exit 2 ;;
esac
if [[ $mode == smoke ]]; then
  "$python" -m pytest tests/test_selection_probability_gradient.py -q
else
  "$python" - "$root" <<'PY'
from pathlib import Path
import json,hashlib,sys
root=Path(sys.argv[1]);smoke=root/'joint_gradient_smoke'
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((smoke/'protocol.json').read_text());r=json.loads((smoke/'result.json').read_text())
if r['smoke'] is not True or r['optimizer_steps']!=0 or r['protocol_sha256']!=digest(smoke/'protocol.json'):
    raise RuntimeError('completed read-only smoke required')
for key,path in [('script_sha256','scripts/audit_joint_guard_probability_gradient.py'),
                 ('mass_helper_sha256','scripts/guard_selection_mass.py'),
                 ('gradient_helper_sha256','scripts/selection_probability_gradient.py')]:
    if p[key]!=digest(path):raise RuntimeError('smoke source differs')
print('JOINT_GRADIENT_SMOKE_VERIFIED',flush=True)
PY
fi
exec "$python" -u -m scripts.audit_joint_guard_probability_gradient \
  --domain-root "$cache/harmonized_secondary_20260917_v1/domain" \
  --flow "$cache/harmonized_secondary_20260917_v1/refinement/selected.pt" \
  --reference-root "$reference" --output-root "$root/joint_gradient_$mode" "${options[@]}"
