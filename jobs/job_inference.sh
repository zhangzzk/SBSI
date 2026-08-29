#!/usr/bin/env bash
# Reference Slurm launcher for v1.1-infer with v3.2-like.

#SBATCH --job-name=sbsi_inference
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}

: "${OUTPUT:?set OUTPUT to a new output directory}"
: "${SCENE_STORE:?set SCENE_STORE to a prepared scene store}"
: "${FLOW:?set FLOW to the v3.2-like measurement-flow checkpoint}"
: "${MODEL_CACHE:?set MODEL_CACHE to the catalogue-model cache}"
: "${PROPOSAL_CACHE:?set PROPOSAL_CACHE to the proposal-coordinate cache}"

args=(
  --inference-config "$repo/configs/inference.json"
  --likelihood-config "$repo/configs/likelihood.json"
  --scene-store "$SCENE_STORE"
  --measurement-model "$FLOW"
  --model-cache "$MODEL_CACHE"
  --proposal-cache "$PROPOSAL_CACHE"
  --output "$OUTPUT"
  --observation-start "${OBSERVATION_START:-0}"
  --device "${DEVICE:-cuda}"
)

if [[ -n "${MOCK_INPUT:-}" ]]; then
  args+=(--mock-input "$MOCK_INPUT")
else
  args+=(
    --n-detected "${N_DETECTED:-10000}"
    --injected-g1 "${INJECTED_G1:-0.0}"
    --injected-g2 "${INJECTED_G2:-0.0}"
    --scene-seed "${SCENE_SEED:-1001}"
    --detection-seed "${DETECTION_SEED:-2001}"
    --flow-seed "${FLOW_SEED:-3001}"
  )
fi

if [[ -n "${OBSERVATION_STOP:-}" ]]; then
  args+=(--observation-stop "$OBSERVATION_STOP")
fi
if [[ -n "${BLEND_RESPONSE_CACHE:-}" ]]; then
  args+=(--blend-response-cache "$BLEND_RESPONSE_CACHE")
fi
if [[ -n "${CUT_ABS_EHAT:-}" ]]; then
  args+=(--cut-abs-ehat "$CUT_ABS_EHAT")
fi
if [[ -n "${CUT_MAG_HI:-}" ]]; then
  args+=(--cut-bound "measured_mag_auto::$CUT_MAG_HI")
fi
if [[ -n "${CUT_LOG_RADIUS_LO:-}" ]]; then
  args+=(--cut-bound "measured_log_flux_radius:$CUT_LOG_RADIUS_LO:")
fi
if [[ -n "${CUT_ABS_EHAT:-}${CUT_MAG_HI:-}${CUT_LOG_RADIUS_LO:-}" ]]; then
  : "${SELECTION_CACHE:?set SELECTION_CACHE when measured cuts are enabled}"
  args+=(--selection-cache "$SELECTION_CACHE")
fi
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
  args+=(--prepare-only)
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
exec "$python" "$repo/scripts/run_inference.py" "${args[@]}"
