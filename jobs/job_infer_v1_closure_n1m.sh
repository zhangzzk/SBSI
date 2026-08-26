#!/usr/bin/env bash
# Infer two disjoint 500k partitions concurrently on two V100s.

#SBATCH --job-name=sbsi_infer_v1_n1m
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=224G
#SBATCH --time=10:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_fs2_n1m_closure_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_fs2_n1m_closure_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/infer_v1_fs2_n1m_closure_v1
compact="$source_root/compact_global"

run_partition() {
  local gpu=$1
  local start=$2
  local stop=$3
  local tag=$4
  CUDA_VISIBLE_DEVICES="$gpu" env \
    OUTPUT="$root/results/$tag" \
    MOCK_INPUT="$root/prepare/mock" \
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
    ADAPTIVE_DRAW_LADDER="512 1024 2048 4096 8192 16384" \
    RETAIN_FULL_LADDER=1 COMPILE_FLOW=1 CANDIDATE_BACKEND=torch \
    OBJECT_CHUNK=128 ATOM_CHUNK=4096 \
    OBSERVATION_START="$start" OBSERVATION_STOP="$stop" \
    OMP_NUM_THREADS=8 \
    bash "$repo/jobs/job_infer_v1.sh" \
    >"$root/logs/infer_${tag}.out" \
    2>"$root/logs/infer_${tag}.err"
}

if [[ -e "$root/results" ]]; then
  echo "refusing to reuse existing results directory $root/results" >&2
  exit 2
fi
mkdir -p "$root/results"

run_partition 0 0 500000 observations_000000_499999 &
first_pid=$!
run_partition 1 500000 1000000 observations_500000_999999 &
second_pid=$!
wait "$first_pid"
wait "$second_pid"

echo "both 500k Infer V1 closure partitions complete"
