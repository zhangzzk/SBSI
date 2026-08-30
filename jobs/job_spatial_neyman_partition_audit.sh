#!/usr/bin/env bash
# Outcome-blind spatial-cell partition audit for the frozen Neyman diagnostic.
# This job builds only catalogue-side geometry and exits before science rows,
# candidate queries, or observed-target flow evaluations are reached.

#SBATCH --job-name=sbsi_spatial_partition_audit
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=02:00:00
# Slurm opens these paths before the job can create an absolute log directory.
# Keep them relative to the directory from which this script is submitted.
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI-inference-sampler-night}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
expected_commit=${SBSI_COMMIT:-}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
root=${DIAGNOSTIC_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/complement_spatial_neyman_center_dev_v1}
compact="$source_root/compact_global"
output=${SPATIAL_NEYMAN_PARTITION_OUTPUT:-$root/partition_audit_v1}

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
if [[ ! -f "$compact/proposal_cache/manifest.json" ]]; then
  echo "missing frozen proposal cache at $compact/proposal_cache" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
mkdir -p "$root" "$root/logs"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

"$python" - <<'PREFLIGHT' || exit 2
from dataclasses import asdict

from sbsi.catalogue_sampling import (
    SPATIAL_PARTITION_RECIPE_SHA256,
    TAIL_PROPOSAL_RECIPE,
)
from sbsi.spatial_neyman_diagnostic import SPATIAL_NEYMAN_PROTOCOL

protocol = SPATIAL_NEYMAN_PROTOCOL
if protocol["n_cells"] != 256:
    raise SystemExit("spatial audit cell count changed")
if protocol["partition_recipe_sha256"] != SPATIAL_PARTITION_RECIPE_SHA256:
    raise SystemExit("spatial audit partition recipe changed")
if protocol["tail_proposal_recipe"] != asdict(TAIL_PROPOSAL_RECIPE):
    raise SystemExit("spatial audit conditioned-DM3 recipe changed")
if protocol["active_atoms"] != 12_760_990:
    raise SystemExit("spatial audit active-support size changed")
PREFLIGHT

exec "$python" "$repo/scripts/run_inference.py" \
  --inference-config "$repo/configs/inference.json" \
  --likelihood-config "$repo/configs/likelihood.json" \
  --scene-store "$compact/scene_store" \
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  --model-cache "$compact/model_cache" \
  --proposal-cache "$compact/proposal_cache" \
  --output "$output" \
  --candidate-backend torch \
  --audit-spatial-neyman-partition \
  --device cuda
