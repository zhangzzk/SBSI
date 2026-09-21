#!/usr/bin/env bash
#SBATCH --job-name=sbsi_fresh_parent
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fresh_fullscene_joint_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fresh_fullscene_joint_20260918_v1/logs/%x_%j.err
set -euo pipefail
mode=${1:?smoke or full}
data=/project/ls-gruen/users/zekang.zhang
root=$data/sbsi_caches/fresh_fullscene_joint_20260918_v1
python=$data/envs/py31/bin/python
models=$data/blendemu_runs/fixed_g0_m258_r060_v2/models
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd /home/z/Zekang.Zhang/SBSI
case $mode in
  smoke)
    output=$root/smoke
    scenes=(--scene "$data/blendemu_runs/joint_multiaxis_pilot_20260918_v1/images" --operational-small-scene)
    cases=(20000) ;;
  full)
    output=$root/full
    scenes=()
    cases=(20010 20011 20012 20013)
    for case_id in "${cases[@]}"; do scenes+=(--scene "$data/blendemu_runs/joint_fullscene_development_20260918_v1/case${case_id}"); done
    "$python" - "$root" <<'PY'
from pathlib import Path
import hashlib, json, sys
root = Path(sys.argv[1])
smoke = json.loads((root/'smoke/preparation.json').read_text())
if (smoke['status'] != 'completed' or smoke['operational_small_scene'] is not True
        or smoke['script_sha256'] != hashlib.sha256(Path('scripts/prepare_fresh_joint_development.py').read_bytes()).hexdigest()):
    raise RuntimeError('matching preparation smoke required')
PY
    ;;
  *) exit 2 ;;
esac
"$python" -m pytest tests/test_fresh_joint_development.py -q
"$python" -u -m scripts.prepare_fresh_joint_development "${scenes[@]}" --output-root "$output"
"$python" -u -m scripts.build_constgold_fixed_g0_blend_lookup \
  --cases "${cases[@]}" \
  --input-pattern "$output/catalogues/g0/case{case}_0.0/real0/catalogues/input/gals_info_tile180.0_-0.5.feather" \
  --anchor-pattern "$output/parent/case{case:03d}.npz" \
  --measurement-model "$data/sbsi_caches/harmonized_secondary_20260917_v1/refinement/selected.pt" \
  --emulator-model "$models/regression_model_lsst_r_fixed_g0_m258_r060_v2.json" \
  --emulator-metadata "$models/emulator_metadata_lsst_r_fixed_g0_m258_r060_v2.json" \
  --output "$output/pair_lookup.feather" --pair-output-root "$output/pairs" --device cpu \
  --population-description 'generated secondary-role true-r<26 parent common to rendered legs; no observed-g0 or measured selection'
if [[ $mode == smoke ]]; then
  "$python" - "$output" "$data/sbsi_caches/joint_multiaxis_evaluator_20260918_v1" <<'PY'
import numpy as np
from pathlib import Path
import sys
current, reference = map(Path, sys.argv[1:])
with np.load(current/'parent/case20000.npz') as new, np.load(reference/'parent/case20000.npz') as old:
    for key in ('input_index', 'context'):
        np.testing.assert_array_equal(new[key], old[key])
print('FRESH_PARENT_SMALL_PILOT_IDENTITY_PASSED', flush=True)
PY
fi
