#!/usr/bin/env bash
#SBATCH --job-name=sbsi_scene_classifier_prep
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
scope=${1:?smoke or full}
case "$scope" in
 smoke) cases=(40 100) ;;
 full)
   mapfile -t cases < <(seq 40 119)
   protocol=${2:?frozen preparation protocol required}
   "$python" - "$protocol" <<'CHECK'
from pathlib import Path
import hashlib,json,sys
p=json.loads(Path(sys.argv[1]).read_text())
for path,expected in p['files'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Geometry preparation input changed: '+path)
print('SCENE_CLASSIFIER_PREPARATION_PROTOCOL_VERIFIED',flush=True)
CHECK
   ;;
 *) exit 2 ;;
esac
"$python" -m pytest tests/test_scene_classifier_features.py -q
exec "$python" -u -m scripts.prepare_scene_classifier \
 --anchor-root "$base/constgold_truth_parent_20260917_v1/parent_v2" \
 --pair-root "$base/output_conditioned_response_20260917_v1/pairs" \
 --domain-root "$base/harmonized_secondary_20260917_v1/domain" \
 --output-root "$base/scene_classifier_geometry_20260918_v1/prepared_${scope}" \
 --cases "${cases[@]}"
