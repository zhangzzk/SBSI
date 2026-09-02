#!/usr/bin/env bash
# The cont.344 bright/faint hybrid, run end to end on every allocated GPU.
#
# cont.342 showed the one-step estimator is biased by its own linearisation and
# that re-solving from the previous estimate converges in three passes.  Three
# full passes cost 3x.  cont.344 measured, by post-processing three completed
# runs, that recomputing only the objects brighter than mag 22 -- 7.91% of them
# -- and carrying the rest by their own linear response
#
#     s_faint(g) = s_faint(g0) - I_faint(g0) (g - g0)
#
# reproduces the converged answer to 0.14 sigma for 1.16 pass-equivalents.
# That was arithmetic on saved moments.  This job is the first time the scheme
# actually runs: passes 1 and 2 evaluate only the bright rows, so the saving is
# real wall clock rather than a projection.
#
# WHAT IT DOES NOT SETTLE.  The chain here is not compared against the exact
# three-pass chain on the same window, because the exact chain was run at
# 25,000 objects and this runs at 20,000.  What it does settle is whether the
# machinery works, what a bright-only pass actually costs including its fixed
# overhead, and whether the answer moves between pass 2 and pass 3.
#
# The carry is always taken from pass 0, never from the previous carried pass,
# so the linear extrapolation is applied once from a genuinely evaluated point.
#
#SBATCH --job-name=sbsi_hybrid_chain
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_hybrid_chain_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_hybrid_chain_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}

source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
source_mock=${SOURCE_MOCK:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_candidate_ladder_n125k_v1/prepare_cont303/mock}
root=${CHAIN_ROOT:-/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_hybrid_chain_v1}
compact="$source_root/compact_global"

N_OBJECTS=${N_OBJECTS:-20000}
MAG_CUT=${MAG_CUT:-22.0}
PASSES=${PASSES:-2}          # recompute passes after pass 0
TAG=${TAG:?set TAG}
BUILD_SELECTION_NORMALIZATION_QUADRATIC=${BUILD_SELECTION_NORMALIZATION_QUADRATIC:-1}
SCORE_ROOT_BFGS=${SCORE_ROOT_BFGS:-0}
INFERENCE_CONFIG=${INFERENCE_CONFIG:-$repo/configs/inference_v1_2.json}

run="$root/$TAG"
# RESUME=1 keeps completed stages.  A pass over the full window costs real GPU
# time and the stages after it are independent, so a failure in a combine step
# should not force the whole chain to be recomputed.  A stage counts as done
# only if it wrote its own result; a directory without one is an interrupted
# stage and is removed.
if [[ -e "$run" && "${RESUME:-0}" != 1 ]]; then
  echo "refusing to overwrite $run (set RESUME=1 to keep completed stages)" >&2
  exit 2
fi

stage_done() {  # stage_done <directory> <result file name>
  [[ -f "$1/$2" ]]
}
clear_incomplete() {
  if [[ -d "$1" ]] && ! stage_done "$1" "$2"; then
    echo "[hybrid] discarding incomplete $1"
    rm -rf "$1"
  fi
}
if [[ ! -f "$source_mock/likelihood_mock_manifest.json" \
      && ! -f "$source_mock/image_mock_manifest.json" ]]; then
  echo "missing provenance-checked frozen mock at $source_mock" >&2; exit 2
fi
mkdir -p "$run" "$root/logs"

if [[ -n "${N_GPUS:-}" ]]; then
  n_gpus=$N_GPUS
elif [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
  n_gpus=$SLURM_GPUS_ON_NODE
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
  n_gpus=${#visible_gpus[@]}
else
  n_gpus=$($python -c 'import torch; print(max(torch.cuda.device_count(), 1))')
fi
if ! [[ "$n_gpus" =~ ^[1-9][0-9]*$ ]] || (( n_gpus > N_OBJECTS )); then
  echo "invalid GPU count $n_gpus for $N_OBJECTS observations" >&2
  exit 2
fi
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-16} / n_gpus ))
if (( OMP_NUM_THREADS < 1 )); then export OMP_NUM_THREADS=1; fi
export PYTHONPATH="$repo:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}${PYTHONPATH:+:$PYTHONPATH}"

"$python" - <<'PREFLIGHT' || exit 2
import sbsi, blendemu  # noqa: F401
PREFLIGHT

common=(
  --inference-config "$INFERENCE_CONFIG"
  --likelihood-config "$repo/configs/likelihood.json"
  --scene-store "$compact/scene_store"
  --measurement-model /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
  --model-cache "$compact/model_cache"
  --proposal-cache "$compact/proposal_cache"
  --mock-input "$source_mock"
  --device cuda
)
if [[ -n "${ADAPTIVE_DRAW_LADDER:-}" ]]; then
  read -r -a draw_ladder <<< "$ADAPTIVE_DRAW_LADDER"
  common+=(--adaptive-draw-ladder "${draw_ladder[@]}")
fi

# Complete-likelihood terms are explicit because the self-response-only mock
# remains a supported diagnostic input.  A supplied normalization cache is
# validated against the scene/model/cut identity.  Otherwise the launcher
# constructs an exact nine-view cache by sharding prior atoms across all GPUs.
if [[ -n "${BLEND_CACHE:-}" ]]; then
  common+=(--blend-response-cache "$BLEND_CACHE")
fi
if [[ -n "${CUT_ABS_EHAT:-}" ]]; then
  common+=(--cut-abs-ehat "$CUT_ABS_EHAT")
fi
if [[ -n "${CUT_BOUND:-}" ]]; then
  for spec in $CUT_BOUND; do
    common+=(--cut-bound "$spec")
  done
fi
if [[ -n "${PROPOSAL_CANDIDATE_SOURCE:-}" ]]; then
  common+=(--proposal-candidate-source "$PROPOSAL_CANDIDATE_SOURCE")
fi

gpu_indices=()
for ((gpu = 0; gpu < n_gpus; gpu++)); do gpu_indices+=("$gpu"); done

wait_all() {
  local status=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then status=1; fi
  done
  return "$status"
}

# Prepare a shared exact B_W stencil.  With multiple GPUs this shards the
# active prior atoms, not the observations, and combines nine scalar partial
# sums.  All subsequent observation shards consume the same validated cache.
normalization_cache=
prepare_normalization() {  # prepare_normalization <pass index> [fixed centre]
  local pass_index=$1 centre=${2:-} shard_dir gpu
  if [[ -n "${SELECTION_NORMALIZATION_CACHE:-}" ]]; then
    normalization_cache=$SELECTION_NORMALIZATION_CACHE
    return
  fi
  if [[ -z "${CUT_ABS_EHAT:-}${CUT_BOUND:-}" ]]; then
    normalization_cache=
    return
  fi
  normalization_cache="$run/pass${pass_index}_selection_normalization.json"
  if [[ -f "$normalization_cache" ]]; then
    echo "[hybrid] reusing exact normalization $normalization_cache"
    return
  fi
  echo "[hybrid] pass $pass_index: exact normalization on $n_gpus GPU atom shards"
  local pids=()
  for gpu in "${gpu_indices[@]}"; do
    shard_dir="$run/pass${pass_index}_normalization_p$gpu"
    clear_incomplete "$shard_dir" selection_normalization_shard.json
    if stage_done "$shard_dir" selection_normalization_shard.json; then
      echo "[hybrid] reusing pass${pass_index}_normalization_p$gpu"
      continue
    fi
    centre_args=()
    if [[ -n "$centre" ]]; then
      centre_args=(--initial-strategy fixed --initial "$centre")
    fi
    CUDA_VISIBLE_DEVICES=$gpu "$python" "$repo/scripts/run_inference.py" \
      "${common[@]}" "${centre_args[@]}" \
      --selection-normalization-shard-index "$gpu" \
      --selection-normalization-shards "$n_gpus" \
      --output "$shard_dir" \
      > "$run/pass${pass_index}_normalization_p$gpu.log" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"
  combine_args=()
  for gpu in "${gpu_indices[@]}"; do
    combine_args+=(--shard "$run/pass${pass_index}_normalization_p$gpu")
  done
  "$python" "$repo/scripts/combine_selection_normalization_shards.py" \
    "${combine_args[@]}" --output "$normalization_cache" \
    > "$run/pass${pass_index}_normalization_combine.log" 2>&1
}

# ---- pass 0: full window, split contiguously across all allocated GPUs -----
prepare_normalization 0
normalization_args=()
if [[ -n "$normalization_cache" ]]; then
  normalization_args=(--selection-normalization-cache "$normalization_cache")
fi
echo "[hybrid] pass 0: $N_OBJECTS objects on $n_gpus GPUs"
for gpu in "${gpu_indices[@]}"; do
  clear_incomplete "$run/pass0_p$gpu" result.json
done
pids=()
for gpu in "${gpu_indices[@]}"; do
  if stage_done "$run/pass0_p$gpu" result.json; then
    echo "[hybrid] reusing pass0_p$gpu"
    continue
  fi
  start=$(( gpu * N_OBJECTS / n_gpus ))
  stop=$(( (gpu + 1) * N_OBJECTS / n_gpus ))
  CUDA_VISIBLE_DEVICES=$gpu "$python" "$repo/scripts/run_inference.py" \
    "${common[@]}" "${normalization_args[@]}" \
    --observation-start "$start" --observation-stop "$stop" \
    --output "$run/pass0_p$gpu" > "$run/pass0_p$gpu.log" 2>&1 &
  pids+=("$!")
done
wait_all "${pids[@]}"

# The window is 20,000 rows of a 500,000-row mock, which the partition combiner
# refuses unless the shortfall is declared.
clear_incomplete "$run/pass0" result.json
if ! stage_done "$run/pass0" result.json; then
  part_args=()
  for gpu in "${gpu_indices[@]}"; do part_args+=(--part "$run/pass0_p$gpu"); done
  "$python" "$repo/scripts/combine_inference_partitions.py" \
    "${part_args[@]}" \
    --allow-partial-window \
    --output "$run/pass0" > "$run/pass0_combine.log" 2>&1
fi

# ---- the bright subset, split evenly across all allocated GPUs -------------
"$python" - "$source_mock" "$N_OBJECTS" "$MAG_CUT" "$run" "$n_gpus" <<'SUBSET'
import sys
import numpy as np
import pandas as pd

mock, n, cut, run, n_gpus = (
    sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5])
)
mag = pd.read_parquet(f"{mock}/measurements.parquet", columns=["measured_mag_auto"])
mag = mag["measured_mag_auto"].to_numpy()[:n]
bright = np.flatnonzero(mag < cut).astype(np.int64)
if bright.size < n_gpus:
    raise SystemExit(f"only {bright.size} objects brighter than {cut}")
# Interleave rather than use contiguous ranges, so the GPUs get equal counts even if the
# bright objects are not uniformly distributed through the mock.
for gpu in range(n_gpus):
    np.save(f"{run}/bright_p{gpu}.npy", bright[gpu::n_gpus])
print(f"bright {bright.size} of {n} ({bright.size / n:.4%}), "
      f"split {[bright[gpu::n_gpus].size for gpu in range(n_gpus)]}")
SUBSET

# ---- recompute passes ------------------------------------------------------
centre=$("$python" -c "
import json,sys
print('%.10f,%.10f' % tuple(json.load(open('$run/pass0/result.json'))['summary']['estimate']))
")
echo "[hybrid] pass 0 estimate: $centre"

for pass_index in $(seq 1 "$PASSES"); do
  echo "[hybrid] pass $pass_index: bright-only at $centre"
  prepare_normalization "$pass_index" "$centre"
  normalization_args=()
  if [[ -n "$normalization_cache" ]]; then
    normalization_args=(--selection-normalization-cache "$normalization_cache")
  fi
  for gpu in "${gpu_indices[@]}"; do
    clear_incomplete "$run/pass${pass_index}_p$gpu" result.json
  done
  pids=()
  for gpu in "${gpu_indices[@]}"; do
    if stage_done "$run/pass${pass_index}_p$gpu" result.json; then
      echo "[hybrid] reusing pass${pass_index}_p$gpu"
      continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu "$python" "$repo/scripts/run_inference.py" \
      "${common[@]}" "${normalization_args[@]}" \
      --object-subset "$run/bright_p$gpu.npy" \
      --allow-indefinite-partition-summary \
      --initial-strategy fixed --initial "$centre" \
      --output "$run/pass${pass_index}_p$gpu" \
      > "$run/pass${pass_index}_p$gpu.log" 2>&1 &
    pids+=("$!")
  done
  wait_all "${pids[@]}"

  base_args=()
  recomputed_args=()
  for gpu in "${gpu_indices[@]}"; do
    base_args+=("$run/pass0_p$gpu")
    recomputed_args+=("$run/pass${pass_index}_p$gpu")
  done
  solver_args=()
  if [[ "$SCORE_ROOT_BFGS" == 1 ]]; then
    solver_args+=(--score-root-bfgs)
    if (( pass_index > 1 )); then
      solver_args+=(--previous "$run/pass$((pass_index - 1))_hybrid.json")
    fi
  fi
  "$python" "$repo/scripts/combine_hybrid_pass.py" \
    --base "${base_args[@]}" \
    --recomputed "${recomputed_args[@]}" \
    "${solver_args[@]}" \
    --output "$run/pass${pass_index}_hybrid.json" \
    > "$run/pass${pass_index}_combine.log" 2>&1

  centre=$("$python" -c "
import json
print('%.10f,%.10f' % tuple(json.load(open('$run/pass${pass_index}_hybrid.json'))['hybrid']['estimate']))
")
  echo "[hybrid] pass $pass_index estimate: $centre"
done

# The exact per-pass stencils are the validation data needed for a reusable
# local quadratic.  Build it automatically after a multi-centre chain so a
# larger catalogue with the same likelihood identity does not repeat this
# population preparation.
if [[ "$BUILD_SELECTION_NORMALIZATION_QUADRATIC" == 1 \
      && -z "${SELECTION_NORMALIZATION_CACHE:-}" \
      && -n "${CUT_ABS_EHAT:-}${CUT_BOUND:-}" \
      && "$PASSES" -ge 1 ]]; then
  quadratic_cache="$run/selection_normalization_quadratic.json"
  validation_args=()
  for pass_index in $(seq 1 "$PASSES"); do
    validation_args+=(--validation-result "$run/pass${pass_index}_p0")
  done
  "$python" "$repo/scripts/build_selection_normalization_cache.py" \
    --source-result "$run/pass0_p0" "${validation_args[@]}" \
    --output "$quadratic_cache" \
    > "$run/selection_normalization_quadratic.log" 2>&1
fi

# ---- chain summary ---------------------------------------------------------
"$python" - "$run" "$PASSES" "$n_gpus" <<'SUMMARY'
import json, os, sys
from pathlib import Path

run, passes, n_gpus = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
chain = []
pass0 = json.loads((run / "pass0" / "result.json").read_text())
chain.append({
    "pass": 0,
    "kind": "full",
    "n_evaluated": pass0["n_observations"],
    "center": pass0["summary"]["center"],
    "estimate": pass0["summary"]["estimate"],
    "robust_standard_error": pass0["summary"]["robust_standard_error"],
    "quadratic_log_likelihood_gain": pass0["summary"]["quadratic_log_likelihood_gain"],
})
for index in range(1, passes + 1):
    payload = json.loads((run / f"pass{index}_hybrid.json").read_text())
    chain.append({
        "pass": index,
        "kind": "bright_only_with_linear_carry",
        "n_evaluated": payload["n_recomputed"],
        "recomputed_fraction": payload["recomputed_fraction"],
        "recomputed_information_share": payload["recomputed_information_share"],
        "center": payload["hybrid"]["center"],
        "estimate": payload["hybrid"]["estimate"],
        "robust_standard_error": payload["hybrid"]["robust_standard_error"],
        "quadratic_log_likelihood_gain": payload["hybrid"]["quadratic_log_likelihood_gain"],
        "carry_only_estimate": payload["carry_only_reference"]["estimate"],
        "hybrid_minus_carry_only": payload["hybrid_minus_carry_only"],
    })
summary = {
    "injected_shear": pass0["injected_shear"],
    "pipeline_release": pass0.get("pipeline_release"),
    "n_objects": pass0["n_observations"],
    "n_gpus": n_gpus,
    "normalization_distribution": "one disjoint active-atom shard per GPU",
    "reusable_normalization_cache": (
        str((run / "selection_normalization_quadratic.json").resolve())
        if (run / "selection_normalization_quadratic.json").is_file()
        else os.environ.get("SELECTION_NORMALIZATION_CACHE")
    ),
    "pass_equivalents": 1.0 + sum(
        entry.get("recomputed_fraction", 0.0) for entry in chain[1:]
    ),
    "chain": chain,
}
(run / "chain.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
SUMMARY

echo "[hybrid] done: $run/chain.json"
