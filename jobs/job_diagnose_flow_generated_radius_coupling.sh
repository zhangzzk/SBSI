#!/usr/bin/env bash
# Test the staged flow's self-response against its own generated g0 radius.

#SBATCH --job-name=sbsi_fgrjoint
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/generated_radius_coupling_c0_19_v2/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/generated_radius_coupling_c0_19_v2/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
v2=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2
output=$v2/generated_radius_coupling_c0_19_v2/result.json

mkdir -p "$(dirname "$output")/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

cases=()
for case in $(seq 0 19); do
  cases+=(--case "$case")
done

"$python" -u -m scripts.diagnose_flow_generated_radius_coupling \
  --domain-root "$v2/domain" \
  --flow "$v2/flow/paired/selected.pt" \
  "${cases[@]}" \
  --output "$output" \
  --draws 64 --sampling-seed 7301 --batch-size 1024 --device cuda

echo "FLOW_GENERATED_RADIUS_JOB_DONE output=$output"
