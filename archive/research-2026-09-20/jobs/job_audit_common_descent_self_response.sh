#!/usr/bin/env bash
#SBATCH --job-name=sbsi_flow_per_leg
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1
harmonized=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
"$python" - "$root" "$harmonized" <<'PY'
from pathlib import Path
import hashlib,json,sys
root,harmonized=map(Path,sys.argv[1:]);digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((root/'model_per_leg_full.json').read_text())
for field,path in [('script_sha256','scripts/audit_flow_per_leg_response.py'),
                   ('label_helper_sha256','scripts/audit_harmonized_usability_response.py'),
                   ('domain_manifest_sha256',harmonized/'domain/manifest.json')]:
    if p[field]!=digest(path):raise RuntimeError('audited estimator or population changed')
if digest(harmonized/'refinement/selected.pt')!='fade8aebfb4825344b8e9e36dfc07fce5357649340995670f1bb284df7068bd4':
    raise RuntimeError('initial corrected flow changed')
r=json.loads((root/'joint_probability_refinement_full/completed.json').read_text())
if (r['best_step']!=2 or r['smoke'] is not False or r['selected_sha256']!='5434cfafb94ecb169b8b6a012752191bb191b82a787b196de8ed6dac2f1501a5'
        or digest(root/'joint_probability_refinement_full/selected.pt')!=r['selected_sha256']):
    raise RuntimeError('selected common-descent checkpoint changed')
print('COMMON_DESCENT_SELF_AUDIT_IDENTITIES_VERIFIED',flush=True)
PY
exec "$python" -u -m scripts.audit_flow_per_leg_response --device cuda \
  --domain-root "$harmonized/domain" --flow "initial=$harmonized/refinement/selected.pt" \
  --flow "common_descent=$root/joint_probability_refinement_full/selected.pt" \
  --output "$root/model_per_leg_common_descent.json"
