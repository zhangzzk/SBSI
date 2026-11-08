#!/usr/bin/env bash
#SBATCH --job-name=sbsi_scene_classifier_train
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
base=/project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1
scope=${1:?smoke or full}
args=()
case "$scope" in
 smoke) args+=(--smoke --device cpu) ;;
 full)
   args+=(--device cuda)
   protocol=${2:?frozen full protocol required}
   "$python" - "$protocol" <<'CHECK'
from pathlib import Path
import hashlib,json,sys
p=json.loads(Path(sys.argv[1]).read_text())
for path,expected in p['files'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Classifier experiment input changed: '+path)
for arm,path in p['smoke_receipts'].items():
    r=json.loads(Path(path).read_text())
    if r['arm']!=arm or r['smoke'] is not True or r['response_read'] is not False:
        raise RuntimeError('Valid arm smoke receipt required')
print('SCENE_CLASSIFIER_FULL_PROTOCOL_VERIFIED',flush=True)
CHECK
   ;;
 *) exit 2 ;;
esac
"$python" -m pytest tests/test_scene_classifier_features.py tests/test_scene_classifier_training.py -q
for arm in control geometry; do
 "$python" -u -m scripts.train_scene_classifier --prepared-root "$base/prepared_${scope}" \
   --output-root "$base/${scope}_${arm}" --arm "$arm" "${args[@]}"
done
