#!/usr/bin/env bash
# Compare the new plain flow and old V3.3-like mean-head flow on identical Re>0.5 support.

#SBATCH --job-name=sbsi_cgmodel_re05
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/constgold/model_compare_re05_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/constgold/model_compare_re05_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python}
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
base=$cache/plain_complete_flow_response_re037_v1
root=$base/constgold/model_compare_re05_v1
classifier=$cache/detection_classifier_transition_lambda1_v1/models/transition_aware.pt

case ${SLURM_ARRAY_TASK_ID:?array task is required} in
  0)
    tag=plain_r10t10_swa
    model=$base/measurement_flow_plain_direct_r10_t10_d16_re037_s501_swaavg.pt
    model_hash=0ca0a9847e17dfb0e15dc1c13df5a633ff3342c45f3e8bf53fb4b3a40c99700b
    ;;
  1)
    tag=meanhead_r500t500_swa
    model=$cache/mixed_shear_e_photometric/measurement_flow_mixed_g0_g005_E_r500_t500_s501_swaavg.pt
    model_hash=4e816ba5cd4be86771008fbbf5f7fec26d3d39cfb2fd90afe52acd0012373d73
    ;;
  *) exit 2 ;;
esac

mkdir -p "$root/logs"
echo "$model_hash  $model" | sha256sum --check --status
echo "9966cfbc191f11b049bf7419dbdb45d65d1262428889a91bb3c9caf928703455  $classifier" | sha256sum --check --status
output=$root/$tag.json
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

cases=()
for case_id in $(seq 40 139); do cases+=(--case "$case_id"); done
rblend=()
for first in $(seq 40 10 130); do
  last=$((first + 9))
  rblend+=(--rblend-lookup "$base/constgold/rblend_re037_complete_c${first}-${last}.feather")
done

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$repo"
"$python" -u -m scripts.evaluate_constgold_complete_forward_response \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --crowd-lookup "$cache/crowd_flux_conc_c0-199.feather" \
  "${rblend[@]}" \
  --measurement-model "$model" \
  --detection-classifier "$classifier" \
  --detection-neighbour-selection impact --detection-impact-exponent 1 \
  "${cases[@]}" \
  --output "$output" \
  --analysis-primary-re-min 0.5 \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 2048 \
  --n-boot 10000 --bootstrap-seed 20260902 --device cuda

echo "CONSTGOLD_RE05_MODEL_COMPARE_DONE tag=$tag output=$output"
