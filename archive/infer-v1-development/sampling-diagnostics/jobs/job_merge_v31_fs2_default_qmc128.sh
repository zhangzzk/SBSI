#!/usr/bin/env bash
# Archived pre-Infer-V1 cache merge.

#SBATCH --job-name=sbsi_fs2_qmc128_merge
#SBATCH --partition=cip
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1

export PYTHONPATH="$repo"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
"$python" "$repo/scripts/merge_sharded_prior_model_qmc.py" \
  --default-prior-manifest /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_prior_cases20000_20199_v1/default_prior_manifest.json \
  --shard-cache-root "$root/shard_caches" \
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  --emulator-metadata "$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json" \
  --emulator-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json \
  --output "$root/compact_global"
