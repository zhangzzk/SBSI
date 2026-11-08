#!/usr/bin/env bash
#SBATCH --job-name=sbsi_guard_normalization
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
mode=${1:?smoke or full}
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
case "$mode" in
  smoke) "$python" -m pytest tests/test_guard_normalization_decomposition.py -q ;;
  full)
    "$python" - "$root" <<'PY'
import hashlib, json, sys
from pathlib import Path
smoke = json.loads((Path(sys.argv[1])/'guard_normalization_smoke.json').read_text())
if smoke['smoke'] is not True or smoke['script_sha256'] != hashlib.sha256(Path('scripts/decompose_guard_normalization.py').read_bytes()).hexdigest():
    raise RuntimeError('matching operational smoke required')
PY
    ;;
  *) exit 2 ;;
esac
exec "$python" -u -m scripts.decompose_guard_normalization \
  --audit "$root/model_per_leg_${mode}.json" --output "$root/guard_normalization_${mode}.json"
