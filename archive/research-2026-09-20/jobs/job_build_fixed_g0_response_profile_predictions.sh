#!/usr/bin/env bash
# Cache staged-flow and v2-emulator response predictions for notebook cases0--19.

#SBATCH --job-name=sbsi_fg0_rspplot
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
blend=/project/ls-gruen/users/zekang.zhang/blendemu_runs
v2=$cache/fixed_g0_m258_r060_v2
output=$v2/response_profile_predictions_c0_19_v2

mkdir -p "$v2/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo:/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

cases=()
for case in $(seq 0 19); do
  cases+=(--case "$case")
done

model_root=$blend/fixed_g0_m258_r060_v2/models
"$python" -u -m scripts.build_fixed_g0_response_profile_predictions \
  --domain-root "$v2/domain" \
  --response-catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather \
  --flow "$v2/flow/paired/selected.pt" \
  --emulator-model "$model_root/regression_model_lsst_r_fixed_g0_m258_r060_v2.json" \
  --emulator-metadata "$model_root/emulator_metadata_lsst_r_fixed_g0_m258_r060_v2.json" \
  "${cases[@]}" --output-root "$output" \
  --draws 64 --sampling-seed 7301 --batch-size 1024 --device cuda

echo "RESPONSE_PROFILE_PREDICTION_JOB_DONE output=$output"
