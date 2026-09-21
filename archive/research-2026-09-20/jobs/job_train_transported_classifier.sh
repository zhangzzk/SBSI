#!/usr/bin/env bash
#SBATCH --job-name=sbsi_transported_classifier
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
scope=${1:?smoke or full}
protocol=${2:?frozen training protocol required}
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1
args=()
case "$scope" in
 smoke) args+=(--smoke) ;;
 full) test -f "$root/transported_classifier_smoke/completed.json" ;;
 *) exit 2 ;;
esac
"$python" - "$protocol" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text())
for path, expected in p['files'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Frozen training input changed: '+path)
print('TRANSPORTED_CLASSIFIER_PROTOCOL_VERIFIED',flush=True)
PY
exec "$python" -u -m scripts.train_transported_classifier \
 --prepared-root "$root/transported_prepared_${scope}" \
 --output-root "$root/transported_classifier_${scope}" \
 --arm geometry --device cpu "${args[@]}"
