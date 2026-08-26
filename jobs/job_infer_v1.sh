#!/usr/bin/env bash
# Infer V1: one-step two-component catalogue inference.

#SBATCH --job-name=sbsi_infer_v1
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1

: "${OUTPUT:?set OUTPUT to a new output directory}"
: "${SCENE_STORE:=$root/scene_store}"
: "${MODEL_CACHE:=$root/model_cache_section5_v1}"
: "${PROPOSAL_CACHE:=$root/proposal_cache_section5_v1}"
: "${FLOW:?set FLOW to the explicit measurement-flow checkpoint (for example V3.2)}"
: "${EMULATOR_METADATA:=$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json}"
: "${EMULATOR_MODEL:=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json}"
: "${MOCK_INPUT:=}"
: "${BLEND_RESPONSE_CACHE:=}"
: "${SELECTION_CACHE:=}"
: "${CUT_ABS_EHAT:=}"
: "${CUT_MAG_HI:=}"
: "${CUT_LOG_RADIUS_LO:=}"
: "${SELECTION_SAMPLES:=64}"
: "${SELECTION_SEED:=8101}"
: "${SELECTION_ROW_CHUNK:=8192}"
: "${N_DETECTED:=10000}"
: "${INJECTED_G1:=0.0}"
: "${INJECTED_G2:=0.0}"
: "${SCENE_SEED:=1001}"
: "${DETECTION_SEED:=2001}"
: "${FLOW_SEED:=3001}"
: "${PREPARE_ONLY:=0}"
: "${OBSERVATION_START:=0}"
: "${OBSERVATION_STOP:=}"
: "${PROPOSAL_FLOW_SAMPLES:=128}"
: "${PROPOSAL_STATISTIC:=mean}"
: "${PROPOSAL_DISPERSION_STATISTIC:=std}"
: "${PROPOSAL_ROW_CHUNK:=8192}"
: "${PROPOSAL_COORDINATE_SEED:=8201}"
: "${DRAWS:=16384}"
: "${PROPOSAL_CANDIDATES:=16384}"
: "${PROPOSAL_PREFILTER_CANDIDATES:=131072}"
: "${PROPOSAL_SEED:=8701}"
: "${PROPOSAL_EPSILON:=0.1}"
: "${PROPOSAL_METHOD:=initial_center_posterior_adapted}"
: "${H:=0.001}"
: "${MAX_ITERATIONS:=10}"
: "${MAX_STEP:=0.01}"
: "${INITIAL:=0,0}"
: "${INITIAL_STRATEGY:=mean_observed_shape}"
: "${MEAN_SHAPE_OFFSET:=0,0}"
: "${MEAN_SHAPE_RESPONSE:=1,0,0,1}"
: "${TOLERANCE:=0.0001}"
: "${MAX_BACKTRACKS:=8}"
: "${COMPILE_FLOW:=1}"
: "${ADAPTIVE_ONE_STEP:=1}"
: "${STRATIFIED_ONE_STEP:=0}"
: "${ADAPTIVE_DRAW_LADDER:=512 1024 2048 4096 8192 16384}"
: "${COMPLEMENT_DRAW_LADDER:=128 256 512}"
: "${ADAPTIVE_MIN_ESS:=32}"
: "${ADAPTIVE_MAX_WEIGHT_FRACTION:=0.5}"
: "${ADAPTIVE_ALLOCATION:=production_prefix}"
: "${ADAPTIVE_PILOT_DRAWS:=512}"
: "${ADAPTIVE_PILOT_SEED:=18701}"
: "${ADAPTIVE_PILOT_SAFETY_FACTOR:=1.0}"
: "${ADAPTIVE_BIAS_CORRECTION:=none}"
: "${CANDIDATE_BACKEND:=scipy}"
: "${RETAIN_FULL_LADDER:=1}"
: "${OBJECT_CHUNK:=128}"
: "${ATOM_CHUNK:=4096}"

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

extra_args=()
if [[ -n "$MOCK_INPUT" ]]; then
  extra_args+=(--mock-input "$MOCK_INPUT")
else
  extra_args+=(
    --n-detected "$N_DETECTED"
    --injected-g1 "$INJECTED_G1" --injected-g2 "$INJECTED_G2"
    --scene-seed "$SCENE_SEED" --detection-seed "$DETECTION_SEED"
    --flow-seed "$FLOW_SEED"
  )
fi
if [[ -n "$BLEND_RESPONSE_CACHE" ]]; then
  extra_args+=(--blend-response-cache "$BLEND_RESPONSE_CACHE")
fi
if [[ -n "$CUT_ABS_EHAT" ]]; then
  extra_args+=(--cut-abs-ehat "$CUT_ABS_EHAT")
fi
if [[ -n "$CUT_MAG_HI" ]]; then
  extra_args+=(--cut-bound "measured_mag_auto::$CUT_MAG_HI")
fi
if [[ -n "$CUT_LOG_RADIUS_LO" ]]; then
  extra_args+=(--cut-bound "measured_log_flux_radius:$CUT_LOG_RADIUS_LO:")
fi
if [[ -n "$CUT_ABS_EHAT$CUT_MAG_HI$CUT_LOG_RADIUS_LO" ]]; then
  : "${SELECTION_CACHE:?set SELECTION_CACHE when measured cuts are enabled}"
  extra_args+=(
    --selection-cache "$SELECTION_CACHE"
    --selection-samples "$SELECTION_SAMPLES"
    --selection-seed "$SELECTION_SEED"
    --selection-row-chunk "$SELECTION_ROW_CHUNK"
  )
fi
if [[ "$COMPILE_FLOW" == 1 ]]; then
  extra_args+=(--compile-flow)
fi
if [[ "$PREPARE_ONLY" == 1 ]]; then
  extra_args+=(--prepare-only)
fi
extra_args+=(--observation-start "$OBSERVATION_START")
if [[ -n "$OBSERVATION_STOP" ]]; then
  extra_args+=(--observation-stop "$OBSERVATION_STOP")
fi
if [[ -n "$PROPOSAL_PREFILTER_CANDIDATES" ]]; then
  extra_args+=(--proposal-prefilter-candidates "$PROPOSAL_PREFILTER_CANDIDATES")
fi
if [[ "$ADAPTIVE_ONE_STEP" == 1 ]]; then
  read -r -a adaptive_draw_ladder <<< "$ADAPTIVE_DRAW_LADDER"
  extra_args+=(
    --adaptive-one-step
    --adaptive-draw-ladder "${adaptive_draw_ladder[@]}"
    --adaptive-min-ess "$ADAPTIVE_MIN_ESS"
    --adaptive-max-weight-fraction "$ADAPTIVE_MAX_WEIGHT_FRACTION"
    --adaptive-allocation "$ADAPTIVE_ALLOCATION"
    --adaptive-pilot-draws "$ADAPTIVE_PILOT_DRAWS"
    --adaptive-pilot-seed "$ADAPTIVE_PILOT_SEED"
    --adaptive-pilot-safety-factor "$ADAPTIVE_PILOT_SAFETY_FACTOR"
    --adaptive-bias-correction "$ADAPTIVE_BIAS_CORRECTION"
    --candidate-backend "$CANDIDATE_BACKEND"
  )
  if [[ "$RETAIN_FULL_LADDER" == 1 ]]; then
    extra_args+=(--retain-full-ladder)
  fi
fi
if [[ "$STRATIFIED_ONE_STEP" == 1 ]]; then
  read -r -a complement_draw_ladder <<< "$COMPLEMENT_DRAW_LADDER"
  extra_args+=(
    --stratified-one-step
    --complement-draw-ladder "${complement_draw_ladder[@]}"
  )
  if [[ "$RETAIN_FULL_LADDER" == 1 ]]; then
    extra_args+=(--retain-full-ladder)
  fi
fi

"$python" "$repo/scripts/run_section5_numerical_recenter.py" \
  --inference-version "Infer V1" \
  --scene-store "$SCENE_STORE" \
  --measurement-model "$FLOW" \
  --emulator-metadata "$EMULATOR_METADATA" --emulator-model "$EMULATOR_MODEL" \
  --model-cache "$MODEL_CACHE" \
  --proposal-cache "$PROPOSAL_CACHE" \
  --proposal-flow-samples "$PROPOSAL_FLOW_SAMPLES" \
  --proposal-statistic "$PROPOSAL_STATISTIC" \
  --proposal-dispersion-statistic "$PROPOSAL_DISPERSION_STATISTIC" \
  --proposal-row-chunk "$PROPOSAL_ROW_CHUNK" \
  --proposal-coordinate-seed "$PROPOSAL_COORDINATE_SEED" \
  --output "$OUTPUT" "${extra_args[@]}" \
  --initial="$INITIAL" --initial-strategy "$INITIAL_STRATEGY" \
  --mean-shape-offset="$MEAN_SHAPE_OFFSET" \
  --mean-shape-response="$MEAN_SHAPE_RESPONSE" \
  --h "$H" --draws "$DRAWS" \
  --proposal-candidates "$PROPOSAL_CANDIDATES" \
  --proposal-seed "$PROPOSAL_SEED" \
  --proposal-epsilon "$PROPOSAL_EPSILON" \
  --proposal-method "$PROPOSAL_METHOD" \
  --max-iterations "$MAX_ITERATIONS" --max-step "$MAX_STEP" \
  --tolerance "$TOLERANCE" --shear-bound 0.1 \
  --max-backtracks "$MAX_BACKTRACKS" \
  --object-chunk "$OBJECT_CHUNK" --atom-chunk "$ATOM_CHUNK" --device cuda
