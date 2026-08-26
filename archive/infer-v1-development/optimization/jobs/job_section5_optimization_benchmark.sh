#!/usr/bin/env bash
# Archived Infer V1 optimization job.
# Compare legacy and optimized Section 5 execution on one frozen null mock.

#SBATCH --job-name=sbsi_s5_opt
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt
metadata="$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json"
emulator=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json

: "${OUTPUT:=$root/optimization_benchmark_n512_k16384_m8192_v1}"
: "${PROPOSAL_CACHE:=$root/proposal_cache_section5_spread_v3}"
: "${N_DETECTED:=512}"
: "${MOCK_G1:=0.0}"
: "${MOCK_G2:=0.0}"
: "${CENTER_G1:=0.0}"
: "${CENTER_G2:=0.0}"
: "${SKIP_LEGACY:=0}"
: "${SKIP_AUTOGRAD:=0}"
: "${RETAIN_FIXED_LADDER:=0}"
: "${PRECISION:=fp32}"
: "${COMPILE_FLOW:=0}"
: "${COMPILE_MODE:=default}"
: "${COMPILE_DYNAMIC:=0}"
: "${DRAW_LADDER:=2048 4096 8192}"
: "${PROPOSAL_CANDIDATES:=16384}"
: "${PROPOSAL_SEED:=8701}"
: "${PROPOSAL_PREFILTER_CANDIDATES:=}"
: "${MIN_ESS:=256}"
: "${MAX_WEIGHT_FRACTION:=0.25}"
: "${ADAPTIVE_ALLOCATION:=production_prefix}"
: "${PILOT_DRAWS:=512}"
: "${PILOT_SEED:=18701}"
: "${PILOT_SAFETY_FACTOR:=1.0}"
: "${ADAPTIVE_BIAS_CORRECTION:=none}"
: "${CANDIDATE_BACKEND:=scipy}"
: "${OBJECT_CHUNK:=128}"
: "${AUTOGRAD_OBJECT_CHUNK:=8}"
: "${ATOM_CHUNK:=4096}"
: "${FULL_INFORMATION:=0}"

optional_args=()
if [[ "$SKIP_LEGACY" == 1 ]]; then optional_args+=(--skip-legacy); fi
if [[ "$SKIP_AUTOGRAD" == 1 ]]; then optional_args+=(--skip-autograd); fi
if [[ "$RETAIN_FIXED_LADDER" == 1 ]]; then optional_args+=(--retain-fixed-ladder); fi
if [[ "$COMPILE_FLOW" == 1 ]]; then optional_args+=(--compile-flow); fi
if [[ "$COMPILE_DYNAMIC" == 1 ]]; then optional_args+=(--compile-dynamic); fi
if [[ "$FULL_INFORMATION" == 1 ]]; then optional_args+=(--full-information); fi
if [[ -n "$PROPOSAL_PREFILTER_CANDIDATES" ]]; then
  optional_args+=(--proposal-prefilter-candidates "$PROPOSAL_PREFILTER_CANDIDATES")
fi
read -r -a draw_ladder <<< "$DRAW_LADDER"

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/benchmark_section5_optimization.py" \
  --scene-store "$root/scene_store" \
  --measurement-model "$flow" \
  --emulator-metadata "$metadata" --emulator-model "$emulator" \
  --model-cache "$root/model_cache_section5_v1" \
  --proposal-cache "$PROPOSAL_CACHE" \
  --output "$OUTPUT" --n-detected "$N_DETECTED" \
  --mock-g1 "$MOCK_G1" --mock-g2 "$MOCK_G2" \
  --center-g1 "$CENTER_G1" --center-g2 "$CENTER_G2" \
  --h 0.005 --draw-ladder "${draw_ladder[@]}" \
  --proposal-candidates "$PROPOSAL_CANDIDATES" \
  --proposal-seed "$PROPOSAL_SEED" --proposal-epsilon 0.1 \
  --min-ess "$MIN_ESS" --max-weight-fraction "$MAX_WEIGHT_FRACTION" \
  --adaptive-allocation "$ADAPTIVE_ALLOCATION" \
  --pilot-draws "$PILOT_DRAWS" --pilot-seed "$PILOT_SEED" \
  --pilot-safety-factor "$PILOT_SAFETY_FACTOR" \
  --adaptive-bias-correction "$ADAPTIVE_BIAS_CORRECTION" \
  --candidate-backend "$CANDIDATE_BACKEND" \
  --precision "$PRECISION" --compile-mode "$COMPILE_MODE" --warmup-rows 65536 \
  --object-chunk "$OBJECT_CHUNK" --autograd-object-chunk "$AUTOGRAD_OBJECT_CHUNK" \
  --atom-chunk "$ATOM_CHUNK" --device cuda \
  "${optional_args[@]}"
