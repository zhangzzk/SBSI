#!/usr/bin/env bash
# One-shot state and log capture for the two fixed-g0 grid continuations.

#SBATCH --job-name=sbsi_fg0grid_check
#SBATCH --partition=cluster,inter
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/logs/%x_%j.err

set -euo pipefail
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2
date --iso-8601=seconds
squeue -j 16524795,16524796 \
  -o '%.20i %.24j %.10T %.10M %.10l %.6D %b %R' || true
sacct -j 16524795,16524796 \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS -n -P || true
for task in 0 1; do
  tail -n 30 "$root/logs/sbsi_fg0grid_16524796_${task}.out" || true
  tail -n 30 "$root/logs/sbsi_fg0grid_16524796_${task}.err" || true
done
find "$root/grid_flows_v1" -path '*/training/completed.json' -print || true
