#!/usr/bin/env bash
# Infer one single-leg FS2 image mock with the finite catalogue prior.

#SBATCH --job-name=sbsi_imgprof
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=52G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/image_closure_fs2_constgold_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/image_closure_fs2_constgold_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
prior=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/realflow_cases0_9_v1
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt
metadata="$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json"
emulator=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json
: "${MOCK_INPUT:=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/image_closure_fs2_constgold_v1/mock_g1p002_c40_44_n2048}"
: "${OUTPUT:=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/image_closure_fs2_constgold_v1/profile_g1p002_c40_44_n2048_v1}"
: "${PROFILE:=-0.01,-0.005,0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05}"
: "${N_DETECTED:=2048}"

if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite $OUTPUT" >&2
  exit 2
fi

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/run_catalogue_closure.py" \
  --scene-store "$prior/scene_store" --measurement-model "$flow" \
  --emulator-metadata "$metadata" --emulator-model "$emulator" \
  --inject-r-blend --blend-response-cache "$prior/blend_response_v1" \
  --output "$OUTPUT" --model-cache "$prior/model_cache_rblend_shapeonly_v1" \
  --proposal-cache "$prior/proposal_cache_rblend_alltargets_v1" \
  --mock-input "$MOCK_INPUT" --sampler importance \
  --profile-shears="$PROFILE" --n-detected "$N_DETECTED" \
  --injected-g1 0.02 --injected-g2 0 --direction-g1 1 --direction-g2 0 \
  --fd-delta 0.005 --no-richardson \
  --proposal-targets measured_ngmix_g1,measured_ngmix_g2,measured_mag_auto,measured_log_flux_radius \
  --proposal-flow-samples 16 --proposal-statistic median \
  --proposal-row-chunk 4096 --proposal-coordinate-seed 7201 \
  --importance-ladder 32768 --proposal-candidates 131072 \
  --proposal-epsilon 0.1 --proposal-bandwidth 1.0 --proposal-seeds 9701 \
  --detection-radius-arcsec 3 --flow-neighbour-radius-arcsec 7 \
  --crowding-near-arcsec 3 --crowding-far-arcsec 7 \
  --pixel-size 0.2 --zero-point 30 --psf-fwhm 0.73 \
  --moffat-beta 2.224 --pixel-rms 0.312 \
  --object-chunk 8 --atom-chunk 2048 --device cuda
