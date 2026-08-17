#!/bin/bash
#SBATCH --job-name=v22g5_concat
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22g5_concat_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_g005_c140-239
SEEDS=(501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517)
"$PY" -u scripts/concat_response_shards.py \
  --shard-dir "$RUN/shards" --output-dir "$RUN/dumps" \
  --tag ablate_s2c_lt500_v22 --seeds "${SEEDS[@]}" \
  --min-case 140 --max-case 240 --cases-per-shard 10
echo V22_G005_CONCAT_DONE
date
