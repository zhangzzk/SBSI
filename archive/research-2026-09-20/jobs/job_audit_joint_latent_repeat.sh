#!/usr/bin/env bash
# Independent latent stream for one frozen model; never select the better seed.
#SBATCH --job-name=sbsi_joint_latent_repeat
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/disk_response_moment_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/disk_response_moment_20260918_v1/logs/%x_%j.err
set -euo pipefail
data=/project/ls-gruen/users/zekang.zhang
cache=$data/sbsi_caches
root=$cache/disk_response_moment_20260918_v1
parent=$cache/constgold_truth_parent_20260917_v1
python=$data/envs/py31/bin/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
mapfile -t cases < <(seq 40 119)
options=()
for seed in 20260913 20260914 20260915; do
  options+=(--classifier "$cache/fixed_g0_m258_r060_v1/classifier/seed${seed}/selected.pt")
done
for first in $(seq 40 10 110); do
  options+=(--rblend "$parent/rblend/cases${first}_$((first+9)).feather")
done
"$python" -u -m scripts.evaluate_constgold_truth_parent \
  --parent-root "$parent/parent_v2" --constgold-root "$data/lsst_sims_fs2_25876_constant" \
  --flow "$cache/full_domain_true_mag26_guard_v1/flow/guard/selected.pt" \
  --response-candidate "$data/blendemu_runs/disk_response_moment_20260918_v1/model" \
  --response-composition mobius --pair-root "$cache/output_conditioned_response_20260917_v1/pairs" \
  --response-threads 8 --response-device cuda --seed 7302 --draws 64 --batch-size 1024 \
  "${options[@]}" --cases "${cases[@]}" --output "$root/joint_latent_seed7302.json"
"$python" -u -m scripts.compare_truth_parent_response \
  --baseline "$root/joint_full.json" --candidate "$root/joint_latent_seed7302.json" \
  --independent-latent-repeat --output "$root/joint_latent_repeat_comparison.json"
