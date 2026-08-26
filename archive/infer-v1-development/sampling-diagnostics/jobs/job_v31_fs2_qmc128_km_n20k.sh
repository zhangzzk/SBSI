#!/usr/bin/env bash
# Archived Infer V1 sampling diagnostic job.
# Run one phase of the paired 20k K x M diagnostic.
#
# PHASE=large and PHASE=small are intended as independent one-GPU jobs on a
# V100 and A40 respectively.  Submit PHASE=finish after both succeed; it runs
# the remaining sequential comparison and selected second-seed check.

#SBATCH --job-name=sbsi_fs2_km20k
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_km_n20k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_km_n20k_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_km_n20k_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python

if [[ -f "$root/candidate_capture_kref131072/result.json" ]]; then
  prefilter=131072
elif [[ -f "$root/candidate_capture_kref65536/result.json" ]]; then
  prefilter=65536
else
  echo "missing completed candidate-capture audit" >&2
  exit 2
fi

run_arm() {
  local gpu=$1
  local k=$2
  local seed=$3
  local output="$root/results/k${k}_e0p1_p${seed}_m512_65536"
  CUDA_VISIBLE_DEVICES="$gpu" env \
    OUTPUT="$output" \
    MOCK_INPUT="$source_root/prepare/mock" \
    SCENE_STORE="$source_root/compact_global/scene_store" \
    MODEL_CACHE="$source_root/compact_global/model_cache" \
    PROPOSAL_CACHE="$source_root/compact_global/proposal_cache" \
    FLOW=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
    BLEND_RESPONSE_CACHE= \
    PROPOSAL_FLOW_SAMPLES=128 \
    PROPOSAL_STATISTIC=mean PROPOSAL_DISPERSION_STATISTIC=std \
    PROPOSAL_COORDINATE_SEED=8201 \
    PROPOSAL_CANDIDATES="$k" PROPOSAL_PREFILTER_CANDIDATES="$prefilter" \
    PROPOSAL_EPSILON=0.1 PROPOSAL_SEED="$seed" \
    PROPOSAL_METHOD=initial_center_posterior_adapted \
    INITIAL_STRATEGY=mean_observed_shape ADAPTIVE_ONE_STEP=1 \
    ADAPTIVE_DRAW_LADDER="512 1024 2048 4096 8192 16384 32768 65536" \
    RETAIN_FULL_LADDER=1 COMPILE_FLOW=1 OBJECT_CHUNK=128 ATOM_CHUNK=4096 \
    OBSERVATION_START=0 OBSERVATION_STOP=20000 \
    OMP_NUM_THREADS=8 \
    bash "$repo/jobs/job_infer_v1.sh" \
    >"$root/logs/k${k}_p${seed}.out" 2>"$root/logs/k${k}_p${seed}.err"
}

mkdir -p "$root/results"

phase=${PHASE:-all}
case "$phase" in
  large)
    run_arm 0 65536 8701
    echo "large-K first-stage arm complete; reference=$prefilter"
    ;;
  small)
    run_arm 0 8192 8701
    echo "small-K first-stage arm complete; reference=$prefilter"
    ;;
  finish)
    [[ -f "$root/results/k65536_e0p1_p8701_m512_65536/result.json" ]]
    [[ -f "$root/results/k8192_e0p1_p8701_m512_65536/result.json" ]]
    run_arm 0 32768 8701
    "$python" "$repo/scripts/summarize_km_diagnostic.py" \
      --root "$root" --mode select --tolerance 0.0001 \
      >"$root/logs/first_stage_summary.out"
    selected_k=$(<"$root/selected_k.txt")
    run_arm 0 "$selected_k" 8702
    "$python" "$repo/scripts/summarize_km_diagnostic.py" \
      --root "$root" --mode final --tolerance 0.0001 \
      >"$root/logs/final_summary.out"
    echo "paired K x M diagnostic complete; reference=$prefilter selected_k=$selected_k"
    ;;
  all)
    run_arm 0 65536 8701
    run_arm 0 8192 8701
    PHASE=finish bash "$0"
    ;;
  *)
    echo "unknown PHASE=$phase (expected large, small, finish, or all)" >&2
    exit 2
    ;;
esac
