#!/usr/bin/env bash
# Run one disjoint partition of the paired 16,384-draw 500k Infer V1 ladder.
# Submit with START, STOP, and TAG exported and a typed GPU on the sbatch line.

#SBATCH --job-name=sbsi_full16384_n500k
#SBATCH --partition=inter
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_full16384_n500k_a40_v100_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_full16384_n500k_a40_v100_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
source_mock=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_stacked_n500k_a40_v100_v1/prepare/mock
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_full16384_n500k_a40_v100_v1
compact="$source_root/compact_global"

: "${START:?set the inclusive observation start}"
: "${STOP:?set the exclusive observation stop}"
: "${TAG:?set the partition output tag}"

output="$root/results/$TAG"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
mkdir -p "$root/results"

exec env \
  OUTPUT="$output" \
  MOCK_INPUT="$source_mock" \
  SCENE_STORE="$compact/scene_store" \
  MODEL_CACHE="$compact/model_cache" \
  PROPOSAL_CACHE="$compact/proposal_cache" \
  FLOW=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  BLEND_RESPONSE_CACHE= \
  PROPOSAL_FLOW_SAMPLES=128 \
  PROPOSAL_STATISTIC=mean PROPOSAL_DISPERSION_STATISTIC=std \
  PROPOSAL_COORDINATE_SEED=8201 \
  PROPOSAL_CANDIDATES=16384 PROPOSAL_PREFILTER_CANDIDATES=131072 \
  PROPOSAL_EPSILON=0.1 PROPOSAL_SEED=8701 \
  PROPOSAL_METHOD=initial_center_posterior_adapted \
  INITIAL_STRATEGY=mean_observed_shape ADAPTIVE_ONE_STEP=1 \
  DRAWS=16384 ADAPTIVE_DRAW_LADDER="512 1024 2048 4096 8192 16384" \
  RETAIN_FULL_LADDER=1 COMPILE_FLOW=1 CANDIDATE_BACKEND=torch \
  OBJECT_CHUNK=128 ATOM_CHUNK=4096 PRECISION=fp32 \
  SBSI_SCORE_PADDED_ATOM_SLOTS=0 \
  OBSERVATION_START="$START" OBSERVATION_STOP="$STOP" \
  bash "$repo/jobs/job_infer_v1.sh"
