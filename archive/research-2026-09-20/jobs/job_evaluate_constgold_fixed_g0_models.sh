#!/usr/bin/env bash
# Compare completed new flows and the frozen fixed-g0-v1 control on ConstGold.

#SBATCH --job-name=sbsi_cgfg0_eval
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1
v1=$cache/fixed_g0_m258_r060_v1
v2=$cache/fixed_g0_m258_r060_v2

mkdir -p "$root/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

cases=()
for case in $(seq 40 89); do
  cases+=(--case "$case")
done
classifiers=()
for seed in 20260913 20260914 20260915; do
  classifiers+=(--classifier "$v1/classifier/seed${seed}/selected.pt")
done

"$python" -u -m scripts.evaluate_constgold_fixed_g0_response \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern "$v2/domain/flow/g0/case{case:03d}.npz" \
  --model "old_fixed_v1_nll=$v1/flow/nll/selected.pt" \
  --model "new_v2_nll=$v2/flow/nll/selected.pt" \
  --model "new_v2_staged=$v2/flow/paired/selected.pt" \
  --model "new_v2_direct=$v2/flow_direct_response/selected.pt" \
  --rblend "old_fixed_v1_nll=$root/rblend_v1_cases40_89.feather" \
  --rblend "new_v2_nll=$root/rblend_v2_cases40_89.feather" \
  --rblend "new_v2_staged=$root/rblend_v2_cases40_89.feather" \
  --rblend "new_v2_direct=$root/rblend_v2_cases40_89.feather" \
  "${classifiers[@]}" "${cases[@]}" \
  --subset cases40_49=40:50 \
  --subset cases40_59=40:60 \
  --subset cases60_79=60:80 \
  --subset cases80_89=80:90 \
  --subset cases40_89=40:90 \
  --output "$root/result.json" \
  --h 0.02 --draws 64 --sampling-seed 7301 --batch-size 1024 \
  --n-boot 10000 --bootstrap-seed 20260914 --device cuda

echo "CONSTGOLD_FIXED_G0_EVALUATION_DONE output=$root/result.json"
