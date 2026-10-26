#!/usr/bin/env bash
#SBATCH --job-name=sbsi_cgparent_eval
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/constgold_truth_parent_20260917_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/constgold_truth_parent_20260917_v1/logs/%x_%j.err
set -euo pipefail
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/constgold_truth_parent_20260917_v1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
mode=${1:-run}
case "$mode" in
  smoke) last=41; lookup_last=40; output=$root/baseline_response_smoke.json ;;
  run) last=119; lookup_last=110; output=$root/baseline_response.json ;;
  *) echo "mode must be smoke or run" >&2; exit 2 ;;
esac
mapfile -t cases < <(seq 40 "$last")
options=()
for seed in 20260913 20260914 20260915; do
  options+=(--classifier "$cache/fixed_g0_m258_r060_v1/classifier/seed${seed}/selected.pt")
done
for first in $(seq 40 10 "$lookup_last"); do
  options+=(--rblend "$root/rblend/cases${first}_$((first+9)).feather")
done
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.evaluate_constgold_truth_parent \
  --parent-root "$root/parent_v2" \
  --constgold-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --flow "$cache/full_domain_true_mag26_guard_v1/flow/guard/selected.pt" \
  "${options[@]}" --output "$output" --cases "${cases[@]}"
