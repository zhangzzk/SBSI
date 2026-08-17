#!/bin/bash
#SBATCH --job-name=v28rb_ana
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v28rb_ana_%j.out
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene/v28_rblend
IFS=',' read -r -a SEEDS <<< "${V28_SEEDS:-501,502}"
OUT=results/v28_rblend_two_seed_screen.json
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }
"$PY" -u scripts/analyze_v27_scene_arms.py \
  --arms rblend --seeds "${SEEDS[@]}" \
  --features-half "$C/rblend_hs_c40-199.feather" \
  --features-const "$C/rblend_const_c40-139.feather" \
  --halfshear-dir "$C/halfshear_scores" \
  --halfshear-baseline-dir /project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v22_c40-199/parts \
  --constgold-dir "$C/constgold_scores" \
  --constgold-baseline-dir /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_domain_shards \
  --output "$OUT"
echo V28_RBLEND_ANALYZE_DONE; date
