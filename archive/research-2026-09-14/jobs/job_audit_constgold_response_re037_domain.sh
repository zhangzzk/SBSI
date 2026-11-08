#!/usr/bin/env bash
# Exact all-case support audit before the Re>0.37 ConstGold response run.

#SBATCH --job-name=sbsi_cgdom_r037
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/constgold/response_c40_139_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/constgold/response_c40_139_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
run=$cache/plain_complete_flow_response_re037_v1/constgold/response_c40_139_v1
flow=$cache/plain_complete_flow_response_re037_v1/measurement_flow_plain_direct_r10_t10_d16_re037_s501_swaavg.pt
emulator_root=/project/ls-gruen/users/zekang.zhang/blendemu_runs/extnbr_reweighted_vector_re037_optuna30_v1/models
emulator=$emulator_root/regression_model_lsst_r_extnbr_re037.json
metadata=$emulator_root/emulator_metadata_lsst_r_extnbr_re037.json

mkdir -p "$run/logs"
[[ ! -e "$run/domain_audit.json" ]] || {
  echo "refusing to overwrite $run/domain_audit.json" >&2
  exit 2
}
cases=()
for case in $(seq 40 139); do cases+=(--case "$case"); done
rblend=()
for first in $(seq 40 10 130); do
  last=$((first + 9))
  rblend+=(--rblend-lookup "$cache/plain_complete_flow_response_re037_v1/constgold/rblend_re037_complete_c${first}-${last}.feather")
done

export PYTHONPATH=$repo
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
"$python" -u "$repo/scripts/audit_constgold_response_domain.py" \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  "${rblend[@]}" \
  --measurement-model "$flow" \
  --emulator-model "$emulator" --emulator-metadata "$metadata" \
  --expected-flow-sha256 0ca0a9847e17dfb0e15dc1c13df5a633ff3342c45f3e8bf53fb4b3a40c99700b \
  --expected-emulator-sha256 167640dd0b876ed54721493c52f3b9fe9412d005b4973e7948566eb3c4001bde \
  "${cases[@]}" \
  --primary-mag-min 18 --primary-mag-max 25.8 \
  --primary-re-min 0.37 --primary-re-max 1.5 \
  --output "$run/domain_audit.json"

echo "CONSTGOLD_RE037_DOMAIN_AUDIT_DONE output=$run/domain_audit.json"
