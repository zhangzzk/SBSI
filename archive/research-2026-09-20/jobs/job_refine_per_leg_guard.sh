#!/usr/bin/env bash
#SBATCH --job-name=sbsi_per_leg_refine
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=38G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/flow_shape_augmentation_20260918_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
"$python" - "$root" <<'PY'
from pathlib import Path
import hashlib, json, sys
root = Path(sys.argv[1])
smoke = root/'per_leg_refinement_mass_smoke_v2'
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
p = json.loads((smoke/'protocol.json').read_text())
r = json.loads((smoke/'completed.json').read_text())
if (r['smoke'] is not True or r['protocol_sha256'] != digest(smoke/'protocol.json')
        or p['script_sha256'] != digest('scripts/refine_per_leg_guard_flow.py')
        or p['selection_mass_helper_sha256'] != digest('scripts/guard_selection_mass.py')
        or p['population_loader_sha256'] != digest('scripts/audit_per_leg_guard_gradient.py')
        or r['selected_sha256'] != digest(smoke/'selected.pt')
        or r['selected_selection_mass']['coordinates'] != ['soft_primary', 'hard_primary']):
    raise RuntimeError('matching completed mass-constrained smoke required')
if p['initial_flow_sha256'] != digest(root.parent/'harmonized_secondary_20260917_v1/refinement/selected.pt'):
    raise RuntimeError('initial corrected flow changed')
print('PER_LEG_FULL_REFINEMENT_SMOKE_VERIFIED', flush=True)
PY
exec "$python" -u -m scripts.refine_per_leg_guard_flow --device cuda \
  --domain-root "$cache/harmonized_secondary_20260917_v1/domain" \
  --flow "$cache/harmonized_secondary_20260917_v1/refinement/selected.pt" \
  --reference-root "$root/per_leg_cv_full" --target-bank-root "$root" \
  --output-root "$root/per_leg_refinement_full"
