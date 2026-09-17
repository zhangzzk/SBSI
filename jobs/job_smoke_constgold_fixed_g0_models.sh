#!/usr/bin/env bash
# One-case CPU smoke test of the fixed-g0 ConstGold evaluator.

#SBATCH --job-name=sbsi_cgfg0_smoke
#SBATCH --partition=cluster,inter
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1
v1=$cache/fixed_g0_m258_r060_v1
v2=$cache/fixed_g0_m258_r060_v2
output=$root/smoke_result.json
[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

"$python" -u -m scripts.evaluate_constgold_fixed_g0_response \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --anchor-pattern "$v2/domain/flow/g0/case{case:03d}.npz" \
  --model "old_fixed_v1_nll=$v1/flow/nll/selected.pt" \
  --model "new_v2_nll=$v2/flow/nll/selected.pt" \
  --rblend "old_fixed_v1_nll=$root/rblend_v1_cases40_89.feather" \
  --rblend "new_v2_nll=$root/rblend_v2_cases40_89.feather" \
  --classifier "$v1/classifier/seed20260913/selected.pt" \
  --classifier "$v1/classifier/seed20260914/selected.pt" \
  --classifier "$v1/classifier/seed20260915/selected.pt" \
  --case 40 --subset case40=40:41 \
  --output "$output" \
  --h 0.02 --draws 2 --sampling-seed 7301 --batch-size 1024 \
  --n-boot 100 --bootstrap-seed 20260914 --device cpu

echo "CONSTGOLD_FIXED_G0_SMOKE_DONE output=$output"
