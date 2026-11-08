#!/usr/bin/env bash
# Verify invariant truth identity across ConstGold +/-0.02 legs.

#SBATCH --job-name=sbsi_cgtruth_audit
#SBATCH --partition=cluster,inter
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1
output=$root/truth_pairing_audit.json
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

cases=()
for case in $(seq 40 89); do
  cases+=(--case "$case")
done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
"$python" -u -m scripts.audit_constgold_truth_pairing \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern /project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/domain/flow/g0/case{case:03d}.npz \
  "${cases[@]}" --h 0.02 --output "$output"

echo "CONSTGOLD_TRUTH_PAIRING_AUDIT_DONE output=$output"
