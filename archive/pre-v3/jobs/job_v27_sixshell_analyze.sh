#!/bin/bash
#SBATCH --job-name=v27sh_ana
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v27sh_ana_%j.out
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene
IFS=',' read -r -a SEEDS <<< "${V27_SEEDS:-501,502}"
OUT=${V27_ANALYSIS_OUT:-results/v27_sixshell_two_seed_screen.json}
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }
"$PY" -u scripts/analyze_v27_scene_arms.py \
  --arms sixshell \
  --seeds "${SEEDS[@]}" \
  --features-half "$C/scene_shells_hs_c0-199.feather" \
  --features-const "$C/scene_shells_const_c40-139.feather" \
  --halfshear-dir "$C/halfshear_scores" \
  --halfshear-baseline-dir /project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v22_c40-199/parts \
  --constgold-dir "$C/constgold_scores" \
  --constgold-baseline-dir /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_domain_shards \
  --output "$OUT"
echo V27_SIXSHELL_ANALYZE_DONE; date
