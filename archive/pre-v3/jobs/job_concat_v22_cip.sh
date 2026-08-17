#!/bin/bash
#SBATCH --job-name=cat22_cip
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/cat22_cip_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

IFS=: read -r -a SEEDS <<< "${V22_SEEDS:-501:502}"
"$PY" -u scripts/concat_response_shards.py \
  --shard-dir /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_domain_shards \
  --output-dir /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps \
  --tag ablate_s2c_lt500_v22 --seeds "${SEEDS[@]}" \
  --min-case 40 --max-case 140 --cases-per-shard 10
echo CAT_V22_CIP_DONE; date
