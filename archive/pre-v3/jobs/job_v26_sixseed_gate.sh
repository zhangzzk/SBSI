#!/bin/bash
#SBATCH --job-name=v26sixgate
#SBATCH --time=01:10:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/v26sixgate_%j.out
set -euo pipefail

cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
SEEDS=(501 502 503 505 506 507)

# Slurm rejects dependencies on surviving elements of a partially cancelled array.  Poll only for
# the exact immutable SWA products required by the six-seed pilot, then release the scoring array.
for attempt in $(seq 1 120); do
  missing=0
  for seed in "${SEEDS[@]}"; do
    ck=$CACHE/measurement_flow_g0_ngmix_ablate_s2c_lt500_v26_scene_target_s${seed}_swaavg.pt
    [ -s "$ck" ] || missing=$((missing + 1))
  done
  if [ "$missing" -eq 0 ]; then
    echo "V26_SIXSEED_GATE_PASS seeds=${SEEDS[*]}"; date
    exit 0
  fi
  echo "attempt=$attempt missing=$missing; waiting 30s"; date
  sleep 30
done
echo "V26_SIXSEED_GATE_FAIL: checkpoints still missing after 60 minutes" >&2
exit 1
