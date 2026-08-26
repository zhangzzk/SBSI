#!/usr/bin/env bash
# Ten-case likelihood profile with a measured-output selection normalization.

#SBATCH --job-name=sbsi_cp_selprof
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=52G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/realflow_cases0_9_v1/logs/selection_profile_%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/realflow_cases0_9_v1/logs/selection_profile_%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/realflow_cases0_9_v1
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt
metadata="$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json"
emulator=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json

: "${RUN_NAME:=selection_abs06_profile_coarse}"
: "${ARM:=plus}"
: "${INJECTED_G1:=0.02}"
: "${INJECTED_G2:=0.0}"
: "${DIRECTION_G1:=1.0}"
: "${DIRECTION_G2:=0.0}"
: "${PROFILE:=0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04}"
: "${LADDER:=8192,32768}"
# Default only when unset: an explicitly empty CUT_ABS disables the shape cut.
: "${CUT_ABS=0.6}"
: "${CUT_BOUND:=}"
: "${SELECTION_SAMPLES:=64}"
: "${SELECTION_SEED:=8101}"
: "${SELECTION_CACHE:=$root/${RUN_NAME}_${ARM}_selection_cache}"
: "${MOCK_INPUT:=}"
: "${N_DETECTED:=2048}"
: "${SCENE_SEED:=1501}"
: "${DETECTION_SEED:=2501}"
: "${FLOW_SEED:=3501}"
: "${PROPOSAL_SEEDS:=4501,4502}"
: "${PROPOSAL_CANDIDATES:=131072}"
: "${OBJECT_CHUNK:=8}"

output="$root/$RUN_NAME/$ARM"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
if [[ -e "$SELECTION_CACHE/manifest.json" ]]; then
  echo "reusing selection cache $SELECTION_CACHE"
elif [[ -e "$SELECTION_CACHE" ]]; then
  echo "selection cache path exists without a manifest: $SELECTION_CACHE" >&2
  exit 2
fi

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cut_args=()
if [[ -n "$CUT_ABS" ]]; then
  cut_args+=(--cut-abs-ehat "$CUT_ABS")
fi
if [[ -n "$CUT_BOUND" ]]; then
  cut_args+=(--cut-bound "$CUT_BOUND")
fi
mock_args=()
if [[ -n "$MOCK_INPUT" ]]; then
  mock_args+=(--mock-input "$MOCK_INPUT")
fi

"$python" "$repo/scripts/run_catalogue_closure.py" \
  --scene-store "$root/scene_store" --measurement-model "$flow" \
  --emulator-metadata "$metadata" --emulator-model "$emulator" \
  --output "$output" --model-cache "$root/model_cache_shapeonly_fd005" \
  --selection-cache "$SELECTION_CACHE" "${cut_args[@]}" \
  --selection-samples "$SELECTION_SAMPLES" --selection-seed "$SELECTION_SEED" \
  --selection-row-chunk 4096 "${mock_args[@]}" \
  --proposal-cache "$root/proposal_cache_alltargets_v2" --sampler importance \
  --profile-shears="$PROFILE" \
  --n-detected "$N_DETECTED" --injected-g1 "$INJECTED_G1" --injected-g2 "$INJECTED_G2" \
  --direction-g1 "$DIRECTION_G1" --direction-g2 "$DIRECTION_G2" \
  --fd-delta 0.005 --no-richardson \
  --proposal-targets measured_ngmix_g1,measured_ngmix_g2,measured_mag_auto,measured_log_flux_radius \
  --proposal-flow-samples 16 --proposal-statistic median \
  --proposal-row-chunk 4096 --proposal-coordinate-seed 7201 \
  --importance-ladder "$LADDER" --proposal-candidates "$PROPOSAL_CANDIDATES" \
  --proposal-epsilon 0.1 --proposal-bandwidth 1.0 --proposal-seeds "$PROPOSAL_SEEDS" \
  --detection-radius-arcsec 3 --flow-neighbour-radius-arcsec 7 \
  --crowding-near-arcsec 3 --crowding-far-arcsec 7 \
  --pixel-size 0.2 --zero-point 30 --psf-fwhm 0.73 \
  --moffat-beta 2.224 --pixel-rms 0.312 \
  --scene-seed "$SCENE_SEED" --detection-seed "$DETECTION_SEED" --flow-seed "$FLOW_SEED" \
  --object-chunk "$OBJECT_CHUNK" --atom-chunk 2048 --device cuda
