#!/usr/bin/env bash
# CPU smoke by default. Frozen physical-density comparisons on existing +/-0.02 legs.
#SBATCH --job-name=sbsi_shape_density
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%A_%a.err
set -euo pipefail
mode=${1:?smoke or full}
arms=(control augmented)
index=${SLURM_ARRAY_TASK_ID:?two-arm array required}
if (( index < 0 || index > 1 )); then exit 2; fi
arm=${arms[$index]}
base=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$base/flow_shape_augmentation_20260918_v1
reference=$base/harmonized_secondary_20260917_v1/refinement/selected.pt
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
case $mode in
  smoke) cases=(40 41); device=cpu ;;
  full) mapfile -t cases < <(seq 40 119); device=cuda ;;
  *) exit 2 ;;
esac
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
"$python" - "$root" "$arm" "$reference" "$mode" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, arm, reference, mode = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
protocol = json.loads((root/'joint_validation_protocol.json').read_text())
expected = protocol['arms'][arm]['selected_sha256']
if (digest(root/arm/'selected.pt') != expected
        or digest(reference) != 'fade8aebfb4825344b8e9e36dfc07fce5357649340995670f1bb284df7068bd4'):
    raise RuntimeError('frozen flow changed')
if mode == 'full':
    smoke = json.loads((root/arm/'transported_density_smoke.json').read_text())
    if (smoke['flow_sha256'] != expected or smoke['reference_flow_sha256'] != digest(reference)
            or smoke['cases'] != [40, 41] or smoke['shears'] != [.02, -.02]
            or smoke['script_sha256'] != digest(Path('scripts/audit_disk_candidate_density.py'))):
        raise RuntimeError('matching physical-density smoke required')
PY
exec "$python" -u -m scripts.audit_disk_candidate_density \
  --flow "$root/$arm/selected.pt" --reference-flow "$reference" \
  --cases "${cases[@]}" --device "$device" --shears 0.02 -0.02 \
  --output "$root/$arm/transported_density_${mode}.json"
