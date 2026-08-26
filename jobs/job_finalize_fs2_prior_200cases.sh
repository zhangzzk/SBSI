#!/usr/bin/env bash
# Audit all scene shards and publish the top-level default-prior manifest.

#SBATCH --job-name=sbsi_fs2p_final
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_prior_cases20000_20199_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_prior_cases20000_20199_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_prior_cases20000_20199_v1
output="$root/default_prior_manifest.json"

export PYTHONPATH="$repo"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/finalize_fs2_sharded_prior.py" \
  --root "$root" --cases 20000-20199 --cases-per-shard 10 --output "$output"
