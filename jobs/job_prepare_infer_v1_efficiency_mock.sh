#!/usr/bin/env bash
# Recreate the frozen Infer V1 efficiency mock under the current implementation.

#SBATCH --job-name=sbsi_infer_v1_prep
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_efficiency_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_efficiency_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_efficiency_v1
compact="$source_root/compact_global"
output="$root/prepare_current"

if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
mkdir -p "$root/logs"

exec env \
  OUTPUT="$output" \
  SCENE_STORE="$compact/scene_store" \
  MODEL_CACHE="$compact/model_cache" \
  PROPOSAL_CACHE="$compact/proposal_cache" \
  FLOW=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  BLEND_RESPONSE_CACHE= \
  PROPOSAL_FLOW_SAMPLES=128 \
  PROPOSAL_STATISTIC=mean PROPOSAL_DISPERSION_STATISTIC=std \
  PROPOSAL_COORDINATE_SEED=8201 \
  N_DETECTED=100000 INJECTED_G1=0.02 INJECTED_G2=0.0 \
  SCENE_SEED=12001 DETECTION_SEED=12002 FLOW_SEED=12003 \
  INITIAL_STRATEGY=mean_observed_shape PREPARE_ONLY=1 COMPILE_FLOW=0 \
  bash "$repo/jobs/job_infer_v1.sh"
