#!/usr/bin/env bash
# Build the fixed R_blend cache for the complete existing FS2 prior.

#SBATCH --job-name=sbsi_fs2_rblend
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_full_cases0_9_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_full_cases0_9_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
: "${OUTPUT_ROOT:=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_full_cases0_9_v1}"
: "${FLOW:=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt}"
: "${EMULATOR_METADATA:=$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json}"
: "${EMULATOR_MODEL:=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json}"

response="$OUTPUT_ROOT/blend_response_v1"
if [[ -e "$response" ]]; then
  echo "refusing to overwrite $response" >&2
  exit 2
fi

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/build_catalogue_blend_response.py" \
  --scene-store "$OUTPUT_ROOT/scene_store" \
  --measurement-model "$FLOW" \
  --emulator-metadata "$EMULATOR_METADATA" --emulator-model "$EMULATOR_MODEL" \
  --output "$response" \
  --pixel-size 0.2 --zero-point 30 --psf-fwhm 0.73 \
  --moffat-beta 2.224 --pixel-rms 0.312 --device cuda
