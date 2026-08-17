#!/bin/bash
#SBATCH --job-name=cat26_cip
#SBATCH --time=00:45:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cat26_cip_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

IFS=: read -r -a SEEDS <<< "${V26_SEEDS:-501:502:503:505:506:507:508:509:510:511:512:513:514:515:516:517}"
"$PY" -u scripts/concat_response_shards.py \
  --shard-dir /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v26_scene_domain_shards \
  --output-dir /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v26_scene_cip_domain_dumps \
  --tag ablate_s2c_lt500_v26_scene_target --seeds "${SEEDS[@]}" \
  --min-case 40 --max-case 140 --cases-per-shard 10
echo CAT_V26_CIP_DONE; date
