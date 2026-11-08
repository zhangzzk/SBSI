#!/usr/bin/env bash
#SBATCH --job-name=sbsi_scene_symmetry
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/harmonized_secondary_20260917_v1/logs/%x_%j.err
set -euo pipefail
data=/project/ls-gruen/users/zekang.zhang
python=$data/envs/py31/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd /home/z/Zekang.Zhang/SBSI
"$python" -m pytest tests/test_scene_prior_shape_symmetry.py -q
exec "$python" -u -m scripts.audit_scene_prior_shape_symmetry \
  --catalogue "$data/lsst_sims/FS2/FS2_25876/FS2_25876_IA_Griffith_UDF_i_25.feather" \
  --output "$data/sbsi_caches/harmonized_secondary_20260917_v1/scene_prior_shape_symmetry.json"
