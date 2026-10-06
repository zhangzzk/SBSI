#!/usr/bin/env bash
#SBATCH --job-name=sbsi_matched_model_full
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%A_%a.err
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
base=/project/ls-gruen/users/zekang.zhang
protocol=${1:?frozen protocol required}
index=${SLURM_ARRAY_TASK_ID:?two-task latent-replica array required}
if (( index < 0 || index > 1 )); then exit 2; fi
seed=$((7301+index))
"$python" - "$protocol" <<'PY'
from pathlib import Path
import hashlib,json,sys
p=json.loads(Path(sys.argv[1]).read_text())
for path,expected in p['files'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Frozen matched-model input changed: '+path)
smoke=json.loads(Path(p['smoke_result']).read_text())
if smoke['status']!='completed' or smoke['scope']!='small_area_operational':
    raise RuntimeError('Completed small-image model replay required')
print('MATCHED_MODEL_FULL_PROTOCOL_VERIFIED',flush=True)
PY
exec "$python" -u -m scripts.evaluate_matched_model_probe \
  --probe-root "$base/blendemu_runs/random_sign_probe_fullscene_20260918_v1/images" \
  --prepared-root "$base/sbsi_caches/fresh_fullscene_joint_20260918_v1/full" \
  --flow "initial=$base/sbsi_caches/harmonized_secondary_20260917_v1/refinement/selected.pt" \
  --flow "common_descent=$base/sbsi_caches/flow_shape_augmentation_20260918_v1/joint_probability_refinement_full/selected.pt" \
  --emulator "$base/blendemu_runs/disk_response_moment_20260918_v1/model" \
  --output "$base/sbsi_caches/flow_shape_augmentation_20260918_v1/matched_model_probe_full_seed${seed}.json" \
  --device cuda --threads 4 --draws 64 --batch-size 1024 --seed "$seed"
