#!/usr/bin/env bash
# cont.347: build the atom-aligned R_blend cache for the inference scene.
#
# The catalogue likelihood has always run with `r_blend.enabled = false`: the
# measurement flow supplies the shape/size/magnitude measurement and the
# classifier supplies the detection probability, but the emulator's blending
# response term is absent, so `R_model = R_flow + R_blend` is currently only
# `R_flow` (doc/CONVENTIONS.md section 1).  This builds the missing term for all
# 12,760,990 atoms of the inference scene so the closure test can be run
# against the complete likelihood.

#SBATCH --job-name=sbsi_blend_cache
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/blend_response_inference_scene_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/blend_response_inference_scene_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
compact="$source_root/compact_global"
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/blend_response_inference_scene_v1
output="$root/blend_response_v1"

mkdir -p "$root/logs"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

"$python" - <<'PREFLIGHT' || exit 2
import sbsi, blendemu  # noqa: F401
PREFLIGHT

# Observing conditions are copied from configs/likelihood.json; the cache is
# only valid for the conditions it was built under and records them itself.
exec "$python" "$repo/scripts/build_catalogue_blend_response.py" \
  --scene-store "$compact/scene_store" \
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  --emulator-model "$repo/models/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json" \
  --emulator-metadata "$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json" \
  --output "$output" \
  --pixel-size 0.2 \
  --zero-point 30.0 \
  --psf-fwhm 0.73 \
  --moffat-beta 2.224 \
  --pixel-rms 0.312 \
  --device cuda
