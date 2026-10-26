#!/usr/bin/env bash
#SBATCH --job-name=sbsi_output_joint
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%A_%a.err
set -euo pipefail
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/output_conditioned_response_20260917_v1
parent=$cache/constgold_truth_parent_20260917_v1
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
models=/project/ls-gruen/users/zekang.zhang/blendemu_runs/output_conditioned_response_20260917_v1/models_cpu
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
mode=${1:?smoke or run}
if [[ $mode == smoke ]]; then
  arm=output_conditioned
  cases=(40 41)
  "$python" - "$parent/rblend/cases40_49.feather" "$root/joint_smoke_rblend.feather" <<'PY'
import sys
import pandas as pd
frame = pd.read_feather(sys.argv[1])
frame.loc[frame['case'].isin([40, 41])].reset_index(drop=True).to_feather(sys.argv[2])
PY
  options=(--rblend "$root/joint_smoke_rblend.feather")
  output=$root/joint_smoke_cuda.json
elif [[ $mode == run ]]; then
  arms=(truth_only output_conditioned)
  arm=${arms[${SLURM_ARRAY_TASK_ID:?}]}
  mapfile -t cases < <(seq 40 119)
  options=()
  for first in $(seq 40 10 110); do
    options+=(--rblend "$parent/rblend/cases${first}_$((first+9)).feather")
  done
  output=$root/joint_${arm}.json
else
  exit 2
fi
for seed in 20260913 20260914 20260915; do
  options+=(--classifier "$cache/fixed_g0_m258_r060_v1/classifier/seed${seed}/selected.pt")
done
"$python" -u -m scripts.evaluate_constgold_truth_parent \
  --parent-root "$parent/parent_v2" \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --flow "$cache/full_domain_true_mag26_guard_v1/flow/guard/selected.pt" \
  --response-candidate "$models/$arm" --pair-root "$root/pairs" --response-threads 8 --response-device cuda \
  "${options[@]}" --output "$output" --cases "${cases[@]}"
if [[ $mode == run ]]; then
  "$python" -u -m scripts.compare_truth_parent_response \
    --baseline "$parent/baseline_response.json" --candidate "$output" \
    --output "$root/joint_${arm}_comparison.json"
fi
