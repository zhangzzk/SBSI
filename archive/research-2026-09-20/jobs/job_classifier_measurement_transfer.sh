#!/usr/bin/env bash
#SBATCH --job-name=sbsi_classifier_transfer
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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
  smoke) args+=(--smoke) ;;
  full)
    protocol=${2:?full execution protocol required}
    "$python" - "$protocol" <<'CHECK'
from pathlib import Path
import hashlib,json,sys
p=json.loads(Path(sys.argv[1]).read_text())
for path, expected in p['files'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Classifier audit input changed: '+path)
r=json.loads(Path(p['smoke_result']).read_text())
if r['smoke'] is not True or r['optimizer_steps']!=0:
    raise RuntimeError('Read-only smoke required')
print('CLASSIFIER_TRANSFER_PROTOCOL_VERIFIED', flush=True)
CHECK
    ;;
  *) exit 2 ;;
esac
for seed in 20260913 20260914 20260915; do
  args+=(--classifier "$base/fixed_g0_m258_r060_v1/classifier/seed$seed/selected.pt")
done
"$python" -m pytest tests/test_classifier_measurement_transfer.py -q
exec "$python" -u -m scripts.audit_classifier_measurement_transfer \
  --domain-root "$base/harmonized_secondary_20260917_v1/domain" \
  --old-prepared "$base/fixed_g0_m258_r060_v1/classifier_prepared" \
  --output "$base/flow_shape_augmentation_20260918_v1/classifier_transfer_${scope}.json" "${args[@]}"
