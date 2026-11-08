#!/usr/bin/env bash
#SBATCH --job-name=sbsi_grid_accumulated
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
set -euo pipefail
unset PYTHONPATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${GRID_SOURCE_ROOT:-/home/z/Zekang.Zhang/SBSI}"
: "${GRID_ACCUM_TRIAL_ROOT:?Set a new trial root with prepared data and source verifier}"
: "${GRID_WEIGHT:?Set the agreed response weight}"
: "${GRID_MODE:?Set sampled or full from diagnostic evidence}"
: "${GRID_NLL_BATCH_SIZE:?Set the effective NLL batch size}"
: "${GRID_CELLS_PER_STEP:?Set cells per update}"
: "${GRID_PAIRS_PER_CELL:?Set galaxy samples per cell, ignored in full mode}"
: "${GRID_DRAWS:?Set antithetic draws per galaxy and group}"
: "${GRID_LR:?Set the trial learning rate}"
train_python=${GRID_TRAIN_PYTHON:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/distribution_response_v1/train_env/bin/python}
audit_python=${GRID_AUDIT_PYTHON:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/distribution_response_v1/audit_env/bin/python}
trial_dir="$GRID_ACCUM_TRIAL_ROOT/$GRID_MODE/lambda$GRID_WEIGHT"
initial_options=()
if [[ -n "${GRID_INITIAL_MODEL:-}" || -n "${GRID_RAW_ROOT:-}" ]]; then
  : "${GRID_INITIAL_MODEL:?Set the completed NLL-only disk checkpoint}"
  : "${GRID_RAW_ROOT:?Set its verified original float64 recovery output}"
  initial_options=(--initial-model "$GRID_INITIAL_MODEL" --raw-root "$GRID_RAW_ROOT")
fi
mkdir -p "$trial_dir"
"$audit_python" "$GRID_ACCUM_TRIAL_ROOT/verify_sources.py"
"$audit_python" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py \
  --output "$trial_dir/resources.json"
"$train_python" -u -m scripts.train_flow_grid_accumulated \
  --root "$GRID_ACCUM_TRIAL_ROOT/prepared" \
  "${initial_options[@]}" \
  --output "$trial_dir/measurement_flow_gridaccum_s501.pt" \
  --weight "$GRID_WEIGHT" --mode "$GRID_MODE" \
  --epochs "${GRID_EPOCHS:-3}" --patience "${GRID_PATIENCE:-15}" \
  --nll-batch-size "$GRID_NLL_BATCH_SIZE" --nll-chunk-size "${GRID_NLL_CHUNK_SIZE:-4096}" \
  --cells-per-step "$GRID_CELLS_PER_STEP" --pairs-per-cell "$GRID_PAIRS_PER_CELL" \
  --draws "$GRID_DRAWS" --response-chunk-size "${GRID_RESPONSE_CHUNK_SIZE:-256}" \
  --validation-draws "${GRID_VALIDATION_DRAWS:-64}" --validation-groups "${GRID_VALIDATION_GROUPS:-4}" \
  --validation-batch-size "${GRID_VALIDATION_BATCH_SIZE:-1024}" \
  --lr "$GRID_LR" --max-grad-norm "${GRID_MAX_GRAD_NORM:-5}" \
  --max-val-nll-increase "${GRID_MAX_VAL_NLL_INCREASE:-0.05}" \
  --max-tail-rate-ratio "${GRID_MAX_TAIL_RATE_RATIO:-4}"
"$audit_python" "$GRID_ACCUM_TRIAL_ROOT/verify_sources.py"
