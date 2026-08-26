#!/usr/bin/env bash
# Archived Infer V1 optimization job.
# Diagnose candidate support on a frozen nonzero-shear likelihood mock.

#SBATCH --job-name=sbsi_s5_ksupport
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
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

: "${OUTPUT:=$root/candidate_support_g005_n1024_k16384_v1}"
: "${MOCK_INPUT:=$root/one_step_calmean_g005_n10000_k16384_ladder512-8192_v3/mock}"
: "${PROPOSAL_CACHE:=$root/proposal_cache_section5_spread_v3}"
: "${N_OBJECTS:=1024}"
: "${CENTER:=0.0548984023 -0.0016381019}"
: "${CANDIDATE_PREFIXES:=128 256 512 1024 2048 4096 8192 16384}"
: "${OBJECT_CHUNK:=16}"
: "${ATOM_CHUNK:=4096}"
: "${COMPILE_FLOW:=1}"
: "${FINAL_CANDIDATES:=}"
: "${PREFILTER_PREFIXES:=}"
: "${DIRECT_UNCERTAINTY_CANDIDATES:=}"
: "${LOCATION_BACKEND:=scipy}"

optional_args=()
if [[ "$COMPILE_FLOW" == 1 ]]; then optional_args+=(--compile-flow); fi
if [[ -n "$FINAL_CANDIDATES" ]]; then
  optional_args+=(--final-candidates "$FINAL_CANDIDATES")
fi
if [[ -n "$PREFILTER_PREFIXES" ]]; then
  read -r -a prefilters <<< "$PREFILTER_PREFIXES"
  optional_args+=(--prefilter-prefixes "${prefilters[@]}")
fi
if [[ -n "$DIRECT_UNCERTAINTY_CANDIDATES" ]]; then
  optional_args+=(--direct-uncertainty-candidates "$DIRECT_UNCERTAINTY_CANDIDATES")
fi
read -r -a center <<< "$CENTER"
read -r -a prefixes <<< "$CANDIDATE_PREFIXES"

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/diagnose_section5_candidate_support.py" \
  --scene-store "$root/scene_store" \
  --measurement-model "$flow" \
  --emulator-metadata "$metadata" --emulator-model "$emulator" \
  --model-cache "$root/model_cache_section5_v1" \
  --proposal-cache "$PROPOSAL_CACHE" \
  --mock-input "$MOCK_INPUT" --output "$OUTPUT" \
  --n-objects "$N_OBJECTS" --center "${center[@]}" \
  --candidate-prefixes "${prefixes[@]}" \
  --location-backend "$LOCATION_BACKEND" \
  --object-chunk "$OBJECT_CHUNK" --atom-chunk "$ATOM_CHUNK" \
  --device cuda "${optional_args[@]}"
