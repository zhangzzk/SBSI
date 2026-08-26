#!/usr/bin/env bash
# Real-model catalogue closure profile with fixed external R_blend enabled.

#SBATCH --job-name=sbsi_cp_rbprof
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=52G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/realflow_cases0_9_v1/logs/rblend_profile_%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/realflow_cases0_9_v1/logs/rblend_profile_%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/realflow_cases0_9_v1
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt
metadata="$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json"
emulator=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json
: "${RUN_NAME:=rblend_profile_g1_v1}"
: "${MOCK_INPUT:=}"
: "${PROFILE:=0.005,0.01,0.015,0.02,0.025,0.03,0.035}"
: "${LADDER:=32768}"
: "${PROPOSAL_SEEDS:=4701}"
: "${N_DETECTED:=1024}"
output="$root/$RUN_NAME"

if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mock_args=()
if [[ -n "$MOCK_INPUT" ]]; then
  mock_args+=(--mock-input "$MOCK_INPUT")
fi

"$python" "$repo/scripts/run_catalogue_closure.py" \
  --scene-store "$root/scene_store" --measurement-model "$flow" \
  --emulator-metadata "$metadata" --emulator-model "$emulator" \
  --inject-r-blend --blend-response-cache "$root/blend_response_v1" \
  --output "$output" --model-cache "$root/model_cache_rblend_shapeonly_v1" \
  --proposal-cache "$root/proposal_cache_rblend_alltargets_v1" \
  --sampler importance --profile-shears="$PROFILE" "${mock_args[@]}" \
  --n-detected "$N_DETECTED" --injected-g1 0.02 --injected-g2 0 \
  --direction-g1 1 --direction-g2 0 --fd-delta 0.005 --no-richardson \
  --proposal-targets measured_ngmix_g1,measured_ngmix_g2,measured_mag_auto,measured_log_flux_radius \
  --proposal-flow-samples 16 --proposal-statistic median \
  --proposal-row-chunk 4096 --proposal-coordinate-seed 7201 \
  --importance-ladder "$LADDER" --proposal-candidates 131072 \
  --proposal-epsilon 0.1 --proposal-bandwidth 1.0 --proposal-seeds "$PROPOSAL_SEEDS" \
  --detection-radius-arcsec 3 --flow-neighbour-radius-arcsec 7 \
  --crowding-near-arcsec 3 --crowding-far-arcsec 7 \
  --pixel-size 0.2 --zero-point 30 --psf-fwhm 0.73 \
  --moffat-beta 2.224 --pixel-rms 0.312 \
  --scene-seed 1701 --detection-seed 2701 --flow-seed 3701 \
  --object-chunk 8 --atom-chunk 2048 --device cuda
