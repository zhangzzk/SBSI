#!/bin/bash
#SBATCH --job-name=sbsi-v36-preparation
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/v36_prepare_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
"$PYTHON" -m pytest tests -q
"$PYTHON" scripts/prepare_constgold_inference_input.py \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --likelihood-config configs/likelihood_v3_6_like.json \
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/joint_probability_refinement_full/selected.pt \
  --output "$RUN_ROOT/input" --sample-size 500000 --seed 20260914 --no-truth-cuts
"$PYTHON" scripts/prepare_disk_prior_shard.py \
  --prior-manifest /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_prior_cases20000_20199_v1/default_prior_manifest.json \
  --shard-cache-root /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1/shard_caches \
  --pairing-metadata /project/ls-gruen/users/zekang.zhang/blendemu_runs/fixed_g0_m258_r060_v2/models/emulator_metadata_lsst_r_fixed_g0_m258_r060_v2.json \
  --shard-index 0 --output "$RUN_ROOT/prior/shard_00"
