#!/usr/bin/env bash
# One-shot two-hour state/log capture for the fixed-g0 ConstGold chain.

#SBATCH --job-name=sbsi_cgfg0_check
#SBATCH --partition=cluster,inter
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_response_c40_89_v1/logs/%x_%j.err

set -euo pipefail
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2
date --iso-8601=seconds
squeue -j 16498286,16498288,16499857,16500170,16500171,16500172,16500221,16500276,16500289 \
  -o '%.18i %.24j %.8T %.10M %.10l %.6D %b %R' || true
sacct -j 16498286,16498288,16499857,16500170,16500171,16500172,16500221,16500276,16500289 \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS -n -P || true
tail -n 30 "$root/logs/sbsi_fg0flow_v2_16498286.out" || true
tail -n 30 "$root/logs/sbsi_fg0direct_v2_16498288.out" || true
tail -n 30 "$root/constgold_response_c40_89_v1/logs/sbsi_cgfg0_eval_16500172.out" || true
tail -n 30 "$root/constgold_response_c40_89_v1/logs/sbsi_cgfg0_eval_16500172.err" || true
tail -n 30 "$root/constgold_response_c40_89_v1/logs/sbsi_cgfg0_review_16500289.out" || true
tail -n 30 "$root/constgold_response_c40_89_v1/logs/sbsi_cgfg0_review_16500289.err" || true
