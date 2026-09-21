#!/usr/bin/env bash
#SBATCH --job-name=sbsi_independent_parent
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/independent_joint_validation_20260919_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/independent_joint_validation_20260919_v1/logs/%x_%j.err
set -euo pipefail
data=/project/ls-gruen/users/zekang.zhang
root=$data/sbsi_caches/independent_joint_validation_20260919_v1
images=$data/blendemu_runs/joint_independent_validation_20260919_v1
models=$data/blendemu_runs/fixed_g0_m258_r060_v2/models
PY=$data/envs/py31/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd /home/z/Zekang.Zhang/SBSI
"$PY" - "$root/preparation_protocol.json" <<'CHECK'
from pathlib import Path
import hashlib,json,sys
p=json.loads(Path(sys.argv[1]).read_text())
for path,expected in p['files'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
        raise ValueError('independent preparation source changed: '+path)
print('INDEPENDENT_PREPARATION_PROTOCOL_VERIFIED',flush=True)
CHECK
"$PY" -m pytest tests/test_independent_joint_validation.py tests/test_fresh_joint_development.py -q
mapfile -t cases < <(seq 22000 22039)
scenes=()
for case_id in "${cases[@]}"; do scenes+=(--scene "$images/case${case_id}"); done
"$PY" -u -m scripts.prepare_independent_joint_validation "${scenes[@]}" \
 --design-protocol "$images/protocol.json" --output-root "$root/prepared"
"$PY" -u -m scripts.build_constgold_fixed_g0_blend_lookup \
 --cases "${cases[@]}" \
 --input-pattern "$root/prepared/catalogues/g0/case{case}_0.0/real0/catalogues/input/gals_info_tile180.0_-0.5.feather" \
 --anchor-pattern "$root/prepared/parent/case{case:03d}.npz" \
 --measurement-model "$data/sbsi_caches/harmonized_secondary_20260917_v1/refinement/selected.pt" \
 --emulator-model "$models/regression_model_lsst_r_fixed_g0_m258_r060_v2.json" \
 --emulator-metadata "$models/emulator_metadata_lsst_r_fixed_g0_m258_r060_v2.json" \
 --output "$root/prepared/pair_lookup.feather" --pair-output-root "$root/prepared/pairs" --device cpu \
 --population-description 'predeclared independent secondary-role true-r<26 parent common to all rendered legs; no observed-g0 or measured selection'
exec "$PY" -u -m scripts.prepare_scene_classifier \
 --anchor-root "$root/prepared/parent" --pair-root "$root/prepared/pairs" \
 --output-root "$root/prepared/geometry" --cases "${cases[@]}"
