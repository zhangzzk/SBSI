#!/usr/bin/env bash
#SBATCH --job-name=sbsi_joint_prob_repeat
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
"$python" -m scripts.check_flow_refinement_repeat_protocol
case $mode in
  smoke) options=(--smoke --device cpu); reference=$root/per_leg_cv_smoke; audit=$root/joint_gradient_smoke ;;
  full) options=(--device cuda); reference=$root/per_leg_cv_full; audit=$root/joint_gradient_full ;;
  *) exit 2 ;;
esac
if [[ $mode == smoke ]]; then
  "$python" -m pytest tests/test_selection_probability_gradient.py tests/test_per_leg_refinement_selection.py tests/test_guard_selection_mass.py -q
else
  "$python" - "$root" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]);smoke=root/'joint_probability_refinement_repeat_smoke'
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((smoke/'protocol.json').read_text());r=json.loads((smoke/'completed.json').read_text())
if (r['smoke'] is not True or r['protocol_sha256']!=digest(smoke/'protocol.json')
        or r['selected_sha256']!=digest(smoke/'selected.pt')):
    raise RuntimeError('completed matching common-descent smoke required')
for key,path in [('script_sha256','scripts/refine_joint_guard_probability_flow_repeat.py'),
                 ('selection_mass_helper_sha256','scripts/guard_selection_mass.py'),
                 ('probability_gradient_helper_sha256','scripts/selection_probability_gradient.py')]:
    if p[key]!=digest(path):raise RuntimeError('smoke source changed')
print('JOINT_PROBABILITY_SMOKE_VERIFIED',flush=True)
PY
fi
exec "$python" -u -m scripts.refine_joint_guard_probability_flow_repeat \
  --domain-root "$cache/harmonized_secondary_20260917_v1/domain" \
  --flow "$cache/harmonized_secondary_20260917_v1/refinement/selected.pt" \
  --reference-root "$reference" --probability-audit-root "$audit" --target-bank-root "$root" \
  --output-root "$root/joint_probability_refinement_repeat_$mode" "${options[@]}"
