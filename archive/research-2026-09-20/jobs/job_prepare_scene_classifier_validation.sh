#!/usr/bin/env bash
#SBATCH --job-name=sbsi_scene_classifier_valprep
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
base=/project/ls-gruen/users/zekang.zhang/sbsi_caches
protocol=${1:?frozen validation preparation protocol}
"$python" - "$protocol" <<'CHECK'
from pathlib import Path
import hashlib,json,sys
p=json.loads(Path(sys.argv[1]).read_text())
for path,expected in p['files'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Geometry preparation input changed: '+path)
print('SCENE_CLASSIFIER_VALIDATION_PROTOCOL_VERIFIED',flush=True)
CHECK
mapfile -t cases < <(seq 200 239)
"$python" -u -m scripts.prepare_scene_classifier \
 --anchor-root "$base/harmonized_secondary_20260917_v1/independent_parent" \
 --pair-root "$base/harmonized_secondary_20260917_v1/independent_pairs" \
 --output-root "$base/scene_classifier_geometry_20260918_v1/validation_independent" --cases "${cases[@]}"
exec "$python" -u -m scripts.prepare_scene_classifier \
 --anchor-root "$base/fresh_fullscene_joint_20260918_v1/full/parent" \
 --pair-root "$base/fresh_fullscene_joint_20260918_v1/full/pairs" \
 --output-root "$base/scene_classifier_geometry_20260918_v1/validation_fresh" --cases 20010 20011 20012 20013
