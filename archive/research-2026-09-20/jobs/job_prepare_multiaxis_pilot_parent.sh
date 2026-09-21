#!/usr/bin/env bash
#SBATCH --job-name=sbsi_multiaxis_parent
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/joint_multiaxis_evaluator_20260918_v1/prepare_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/joint_multiaxis_evaluator_20260918_v1/prepare_%j.err
set -euo pipefail
data=/project/ls-gruen/users/zekang.zhang
root=$data/sbsi_caches/joint_multiaxis_evaluator_20260918_v1
pilot=$data/blendemu_runs/joint_multiaxis_pilot_20260918_v1/images
models=$data/blendemu_runs/fixed_g0_m258_r060_v2/models
python=$data/envs/py31/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd /home/z/Zekang.Zhang/SBSI
"$python" -u -m scripts.prepare_multiaxis_pilot_parent --pilot-root "$pilot" --output-root "$root/parent"
exec "$python" -u -m scripts.build_constgold_fixed_g0_blend_lookup \
  --cases 20000 --input-pattern "$pilot/g0/case{case}_0.0/real0/catalogues/input/gals_info_tile180.0_-0.5.feather" \
  --anchor-pattern "$root/parent/case{case:03d}.npz" \
  --measurement-model "$data/sbsi_caches/harmonized_secondary_20260917_v1/refinement/selected.pt" \
  --emulator-model "$models/regression_model_lsst_r_fixed_g0_m258_r060_v2.json" \
  --emulator-metadata "$models/emulator_metadata_lsst_r_fixed_g0_m258_r060_v2.json" \
  --output "$root/pair_lookup.feather" --pair-output-root "$root/pairs" --device cpu \
  --population-description 'complete generated secondary-role truth parent; strict true r<26; no observed-g0 or measured selection'
