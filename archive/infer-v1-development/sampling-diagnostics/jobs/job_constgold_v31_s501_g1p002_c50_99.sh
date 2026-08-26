#!/usr/bin/env bash
# Archived pre-Infer-V1 ConstGold campaign.
# Infer ten ConstGold +g1 blocks with V3.1 seed 501; submit task 0 before 1-9%2.

#SBATCH --job-name=sbsi_v31_imgrec
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=38G
#SBATCH --time=06:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/constgold_v31_s501_g1p002_c50_99_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/constgold_v31_s501_g1p002_c50_99_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
campaign_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/constgold_v31_s501_g1p002_c50_99_v1
prior_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_full_cases0_9_v1
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt

first_case=$((50 + 5 * SLURM_ARRAY_TASK_ID))
last_case=$((first_case + 4))
block_tag="c${first_case}_${last_case}_n2048"
mock="$campaign_root/mocks/mock_g1p002_${block_tag}"
output="$campaign_root/results/recenter_g1p002_${block_tag}"

if [[ ! -f "$mock/image_mock_manifest.json" ]]; then
  echo "missing prepared image mock $mock" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi

exec env \
  FLOW="$flow" \
  SCENE_STORE="$prior_root/scene_store" \
  MODEL_CACHE="$campaign_root/model_cache_v31_s501" \
  PROPOSAL_CACHE="$campaign_root/proposal_cache_v31_s501" \
  BLEND_RESPONSE_CACHE="$prior_root/blend_response_v1" \
  MOCK_INPUT="$mock" OUTPUT="$output" \
  DRAWS=65536 PROPOSAL_CANDIDATES=65536 \
  PROPOSAL_SEED=8701 PROPOSAL_EPSILON=0.5 \
  PROPOSAL_METHOD=initial_center_posterior_adapted \
  H=0.001 INITIAL=0,0 TOLERANCE=0.0001 MAX_BACKTRACKS=8 \
  COMPILE_FLOW=1 OBJECT_CHUNK=128 ATOM_CHUNK=4096 \
  bash "$repo/jobs/job_infer_v1.sh"
