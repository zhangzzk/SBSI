#!/usr/bin/env bash
# cont.347: build R_blend for the inference scene, shard by shard.
#
# The merged inference scene has no neighbour graph (merge_sharded_prior_model_qmc.py
# writes an empty one deliberately), so the single-scene builder returns zero
# pairs there.  The source shards keep the graph -- 2,864,350,310 directed edges
# across 20 shards -- so the response is built there and concatenated in merge
# order.
#
# SHARD_START / SHARD_STOP select a shard range so the work can be split across
# jobs.  MERGE=1 assembles finished parts into the aligned cache instead.

#SBATCH --job-name=sbsi_blend_shards
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/blend_response_inference_scene_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/blend_response_inference_scene_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
prior_manifest=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_prior_cases20000_20199_v1/default_prior_manifest.json
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/blend_response_inference_scene_v1
output="$root/blend_response_v1"
inference_scene=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1/compact_global/scene_store

mkdir -p "$root/logs"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

"$python" - <<'PREFLIGHT' || exit 2
import sbsi, blendemu  # noqa: F401
PREFLIGHT

args=(
  --prior-manifest "$prior_manifest"
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
  --emulator-model "$repo/models/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json"
  --emulator-metadata "$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json"
  --output "$output"
  --pixel-size 0.2
  --zero-point 30.0
  --psf-fwhm 0.73
  --moffat-beta 2.224
  --pixel-rms 0.312
  --device cuda
)
if [[ "${MERGE:-0}" == 1 ]]; then
  args+=(--merge --inference-scene-store "$inference_scene")
else
  args+=(--shard-start "${SHARD_START:-0}")
  [[ -n "${SHARD_STOP:-}" ]] && args+=(--shard-stop "$SHARD_STOP")
fi

exec "$python" "$repo/scripts/build_sharded_blend_response.py" "${args[@]}"
