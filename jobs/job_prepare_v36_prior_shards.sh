#!/bin/bash
#SBATCH --job-name=sbsi-v36-prior
#SBATCH --partition=cluster
#SBATCH --array=1-19%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/v36_prior_%A_%a.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN_ROOT=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
printf -v SHARD_LABEL '%02d' "$SLURM_ARRAY_TASK_ID"
"$PYTHON" scripts/prepare_disk_prior_shard.py \
  --prior-manifest /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_prior_cases20000_20199_v1/default_prior_manifest.json \
  --shard-cache-root /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1/shard_caches \
  --pairing-metadata /project/ls-gruen/users/zekang.zhang/blendemu_runs/fixed_g0_m258_r060_v2/models/emulator_metadata_lsst_r_fixed_g0_m258_r060_v2.json \
  --shard-index "$SLURM_ARRAY_TASK_ID" --output "$RUN_ROOT/prior/shard_$SHARD_LABEL"
