#!/bin/bash
#SBATCH --job-name=merge_v2_rwv
#SBATCH --time=00:20:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/merge_v2_rwv_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/merge_v2_rwv_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1/constgold_lookup_parts
OUT=results/blend_lookup_v2_reweighted_vector_fixed_c40-139.feather

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
"$PY" -u scripts/merge_v2_reweighted_vector_lookup.py \
  --input-glob "$PARTS/lookup_c*.feather" \
  --expected-shards 10 \
  --min-case 40 \
  --max-case 139 \
  --output "$OUT"
test -s "$OUT"
echo V2_REWEIGHTED_VECTOR_FIXED_LOOKUP_DONE
