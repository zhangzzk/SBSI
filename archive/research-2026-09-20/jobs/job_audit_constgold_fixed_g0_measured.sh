#!/usr/bin/env bash
# Independent measured-side audit for fixed-g0 ConstGold cases40--89.

#SBATCH --job-name=sbsi_cgfg0_meas
#SBATCH --partition=cluster,inter
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1
output=$root/measured_audit.json
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

cases=()
for case in $(seq 40 89); do
  cases+=(--case "$case")
done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
"$python" -u -m scripts.audit_constgold_fixed_g0_measured \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern /project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/domain/flow/g0/case{case:03d}.npz \
  "${cases[@]}" \
  --subset cases40_49=40:50 \
  --subset cases40_59=40:60 \
  --subset cases60_79=60:80 \
  --subset cases80_89=80:90 \
  --subset cases40_89=40:90 \
  --output "$output" --h 0.02 --n-boot 10000 --bootstrap-seed 20260914

echo "CONSTGOLD_FIXED_G0_MEASURED_AUDIT_DONE output=$output"
