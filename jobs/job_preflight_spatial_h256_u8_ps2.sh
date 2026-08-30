#!/usr/bin/env bash
# Outcome-free, constructor-only production-scale A40 preflight for H256-U8-PS2.

#SBATCH --job-name=sbsi_h256_u8_ps2_preflight
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_spatial_neyman_center_dev_v1/logs/slurm-h256-ps-preflight-%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_spatial_neyman_center_dev_v1/logs/slurm-h256-ps-preflight-%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI-inference-sampler-night}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
expected_commit=${SBSI_COMMIT:-}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1/compact_global
partition_cache=${SPATIAL_H256_U8_PS2_PARTITION_CACHE:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_spatial_neyman_center_dev_v1/partition_audit_v1}

if [[ ! -d "$repo/.git" && ! -f "$repo/.git" ]]; then
  echo "missing isolated repository checkout at $repo" >&2
  exit 2
fi
if [[ ! "$expected_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SBSI_COMMIT must pin one full lowercase 40-character commit" >&2
  exit 2
fi
actual_commit=$(git -C "$repo" rev-parse --verify HEAD)
if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "checkout commit $actual_commit does not match SBSI_COMMIT $expected_commit" >&2
  exit 2
fi
if git -C "$repo" symbolic-ref -q HEAD >/dev/null 2>&1; then
  echo "refusing a branch checkout; submit from a detached commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo" status --porcelain --untracked-files=normal)" ]]; then
  echo "refusing a dirty repository checkout at $repo" >&2
  exit 2
fi
if [[ ! -f "$source_root/scene_store/manifest.json" ]]; then
  echo "missing production scene store under $source_root" >&2
  exit 2
fi
if [[ ! -f "$source_root/model_cache/manifest.json" ]]; then
  echo "missing production model cache under $source_root" >&2
  exit 2
fi
if [[ ! -f "$source_root/proposal_cache/manifest.json" ]]; then
  echo "missing production proposal cache under $source_root" >&2
  exit 2
fi
if [[ ! -f "$partition_cache/spatial_partition.npz" || ! -f "$partition_cache/manifest.json" ]]; then
  echo "missing reviewed spatial partition under $partition_cache" >&2
  exit 2
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

exec "$python" "$repo/scripts/preflight_spatial_h256_u8_ps2.py" \
  --scene-store "$source_root/scene_store" \
  --model-cache "$source_root/model_cache" \
  --proposal-cache "$source_root/proposal_cache" \
  --partition-cache "$partition_cache" \
  --device cuda
