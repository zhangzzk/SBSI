#!/usr/bin/env bash
# Outcome-free production-shape A40 preflight for the packed integer bridge.

#SBATCH --job-name=sbsi_packed_bridge_preflight
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_integer_bridge_smc_center_dev_v1/logs/slurm-packed-bridge-preflight-%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_integer_bridge_smc_center_dev_v1/logs/slurm-packed-bridge-preflight-%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI-inference-sampler-night}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
expected_commit=${SBSI_COMMIT:-}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1/compact_global
integer_base_cache=${PACKED_BRIDGE_INTEGER_BASE_CACHE:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_integer_bridge_smc_center_dev_v1/integer_base_v1}
report_root=${PACKED_BRIDGE_PREFLIGHT_REPORT_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_integer_bridge_smc_center_dev_v1/preflight_reports}

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
if [[ -e "$integer_base_cache" && ! -d "$integer_base_cache" ]]; then
  echo "integer-base cache path exists but is not a directory: $integer_base_cache" >&2
  exit 2
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

exec "$python" "$repo/scripts/preflight_packed_integer_bridge_smc.py" \
  --scene-store "$source_root/scene_store" \
  --model-cache "$source_root/model_cache" \
  --proposal-cache "$source_root/proposal_cache" \
  --integer-base-cache "$integer_base_cache" \
  --report "$report_root/preflight-${SLURM_JOB_ID}.json" \
  --device cuda
