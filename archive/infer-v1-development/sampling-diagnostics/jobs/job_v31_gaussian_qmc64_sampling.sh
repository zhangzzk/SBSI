#!/usr/bin/env bash
# Archived pre-Infer-V1 sampling campaign.
# Build a V3.1 Gaussian-moment proposal cache, then scan K/epsilon/seed with
# a retained common-draw ladder. Submit MODE=prepare first; submit array 0-11%2
# afterok on the preparation job.

#SBATCH --job-name=sbsi_v31_qmc64
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-11%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_gaussian_qmc64_sampling_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_gaussian_qmc64_sampling_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_gaussian_qmc64_sampling_v1
campaign_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/constgold_v31_s501_g1p002_c50_99_v1
prior_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_full_cases0_9_v1
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
proposal_cache="$root/proposal_cache_v31_s501_qmc64_mean_std_v4"
mock_root="$root/prepare_mock_qmc64_mean_std_rblend0/mock"
: "${MODE:=matrix}"

common_env=(
  FLOW="$flow"
  SCENE_STORE="$prior_root/scene_store"
  MODEL_CACHE="$campaign_root/model_cache_v31_s501"
  PROPOSAL_CACHE="$proposal_cache"
  # Adaptive Section 5 is currently validated only without external R_blend.
  # Keep this sampling-calibration matrix inside that explicit scope.
  BLEND_RESPONSE_CACHE=
  PROPOSAL_FLOW_SAMPLES=64
  PROPOSAL_STATISTIC=mean
  PROPOSAL_DISPERSION_STATISTIC=std
  PROPOSAL_COORDINATE_SEED=8201
  PROPOSAL_PREFILTER_CANDIDATES=32768
  PROPOSAL_METHOD=initial_center_posterior_adapted
  H=0.001
  INITIAL_STRATEGY=mean_observed_shape
  ADAPTIVE_ONE_STEP=1
  RETAIN_FULL_LADDER=1
  COMPILE_FLOW=1
  OBJECT_CHUNK=128
  ATOM_CHUNK=4096
)

if [[ "$MODE" == prepare ]]; then
  output="$root/prepare_mock_qmc64_mean_std_rblend0"
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite $output" >&2
    exit 2
  fi
  exec env \
    "${common_env[@]}" \
    OUTPUT="$output" N_DETECTED=10000 INJECTED_G1=0.02 INJECTED_G2=0.0 \
    SCENE_SEED=12001 DETECTION_SEED=12002 FLOW_SEED=12003 \
    PROPOSAL_CANDIDATES=4096 PROPOSAL_EPSILON=0.2 PROPOSAL_SEED=8701 \
    ADAPTIVE_DRAW_LADDER=512 \
    bash "$repo/jobs/job_infer_v1.sh"
fi

if [[ "$MODE" != matrix ]]; then
  echo "MODE must be prepare or matrix" >&2
  exit 2
fi
: "${SLURM_ARRAY_TASK_ID:?matrix mode requires an array task}"
if [[ ! -f "$mock_root/likelihood_mock_manifest.json" ]]; then
  echo "missing prepared V3.1 likelihood mock $mock_root" >&2
  exit 2
fi

k_values=(4096 8192)
epsilon_values=(0.1 0.2 0.3)
seed_values=(8701 8702)
k_index=$((SLURM_ARRAY_TASK_ID / 6))
remainder=$((SLURM_ARRAY_TASK_ID % 6))
epsilon_index=$((remainder / 2))
seed_index=$((remainder % 2))
k=${k_values[$k_index]}
epsilon=${epsilon_values[$epsilon_index]}
seed=${seed_values[$seed_index]}
epsilon_tag=${epsilon//./p}
output="$root/results/k${k}_pre32768_e${epsilon_tag}_p${seed}_m512_65536"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi

exec env \
  "${common_env[@]}" \
  MOCK_INPUT="$mock_root" OUTPUT="$output" \
  PROPOSAL_CANDIDATES="$k" PROPOSAL_EPSILON="$epsilon" \
  PROPOSAL_SEED="$seed" \
  ADAPTIVE_DRAW_LADDER="512 1024 2048 4096 8192 16384 32768 65536" \
  bash "$repo/jobs/job_infer_v1.sh"
