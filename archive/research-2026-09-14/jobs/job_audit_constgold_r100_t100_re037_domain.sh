#!/usr/bin/env bash
# Verify exact model/emulator/support compatibility before the 100/100 ConstGold arms.

#SBATCH --job-name=sbsi_cgdom_r100
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/r100_t100_v1/constgold/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1/r100_t100_v1/constgold/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
base=$cache/plain_complete_flow_response_re037_v1
root=$base/r100_t100_v1/constgold
flow=$base/r100_t100_v1/measurement_flow_plain_direct_r100_t100_d16_re037_s501_swaavg.pt
emulator_root=/project/ls-gruen/users/zekang.zhang/blendemu_runs/extnbr_reweighted_vector_re037_optuna30_v1/models
emulator=$emulator_root/regression_model_lsst_r_extnbr_re037.json
metadata=$emulator_root/emulator_metadata_lsst_r_extnbr_re037.json
output=$root/domain_audit.json

mkdir -p "$root/logs"
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
read -r flow_sha _ < <(sha256sum "$flow")
cases=()
for case_id in $(seq 40 139); do cases+=(--case "$case_id"); done
rblend=()
for first in $(seq 40 10 130); do
  last=$((first + 9))
  rblend+=(--rblend-lookup "$base/constgold/rblend_re037_complete_c${first}-${last}.feather")
done

export PYTHONPATH=$repo
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
"$python" -u "$repo/scripts/audit_constgold_response_domain.py" \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  "${rblend[@]}" \
  --measurement-model "$flow" \
  --emulator-model "$emulator" --emulator-metadata "$metadata" \
  --expected-flow-sha256 "$flow_sha" \
  --expected-emulator-sha256 167640dd0b876ed54721493c52f3b9fe9412d005b4973e7948566eb3c4001bde \
  "${cases[@]}" \
  --primary-mag-min 18 --primary-mag-max 25.8 \
  --primary-re-min 0.37 --primary-re-max 1.5 \
  --output "$output"

sha256sum "$output"
echo "CONSTGOLD_R100_T100_RE037_DOMAIN_DONE output=$output"
