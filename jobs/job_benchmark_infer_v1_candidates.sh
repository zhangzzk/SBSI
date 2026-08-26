#!/usr/bin/env bash
# Matched Infer V1 candidate-backend benchmark on the frozen FS2 20k rows.

#SBATCH --job-name=sbsi_infer_v1_candidates
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_efficiency_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_efficiency_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_efficiency_v1
backend=${CANDIDATE_BACKEND:-torch}
n_observations=${N_OBSERVATIONS:-20000}
mock_input=${MOCK_INPUT:-$source_root/prepare/mock}
benchmark_tag=${BENCHMARK_TAG:-original_mock}

case "$backend" in
  scipy|torch) ;;
  *)
    echo "unsupported CANDIDATE_BACKEND=$backend" >&2
    exit 2
    ;;
esac
if ((n_observations <= 0 || n_observations > 20000)); then
  echo "N_OBSERVATIONS must lie in [1, 20000]" >&2
  exit 2
fi
if [[ ! "$benchmark_tag" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "BENCHMARK_TAG contains unsupported characters" >&2
  exit 2
fi

output="$root/results/${benchmark_tag}_${backend}_n${n_observations}"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
mkdir -p "$root/logs" "$root/results"

CUDA_VISIBLE_DEVICES=0 env \
  OUTPUT="$output" \
  MOCK_INPUT="$mock_input" \
  SCENE_STORE="$source_root/compact_global/scene_store" \
  MODEL_CACHE="$source_root/compact_global/model_cache" \
  PROPOSAL_CACHE="$source_root/compact_global/proposal_cache" \
  FLOW=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  BLEND_RESPONSE_CACHE= \
  PROPOSAL_FLOW_SAMPLES=128 \
  PROPOSAL_STATISTIC=mean PROPOSAL_DISPERSION_STATISTIC=std \
  PROPOSAL_COORDINATE_SEED=8201 \
  PROPOSAL_CANDIDATES=16384 PROPOSAL_PREFILTER_CANDIDATES=131072 \
  PROPOSAL_EPSILON=0.1 PROPOSAL_SEED=8701 \
  PROPOSAL_METHOD=initial_center_posterior_adapted \
  INITIAL_STRATEGY=mean_observed_shape ADAPTIVE_ONE_STEP=1 \
  ADAPTIVE_DRAW_LADDER="512 1024 2048 4096 8192 16384" \
  RETAIN_FULL_LADDER=1 COMPILE_FLOW=1 \
  CANDIDATE_BACKEND="$backend" OBJECT_CHUNK=128 ATOM_CHUNK=4096 \
  OBSERVATION_START=0 OBSERVATION_STOP="$n_observations" \
  OMP_NUM_THREADS=8 \
  bash "$repo/jobs/job_infer_v1.sh"

echo "Infer V1 candidate benchmark tag=$benchmark_tag backend=$backend N=$n_observations complete"
