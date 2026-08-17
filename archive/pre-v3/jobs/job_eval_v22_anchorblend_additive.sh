#!/bin/bash
#SBATCH --job-name=v22_abadd
#SBATCH --time=03:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22_abadd_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
DUMPS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps
CAL=results/anchorblend_g005_local10_c200-299.json
OUT=results/v22_anchorblend_additive_constgold_c40-139_s16.json
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
"$PY" -u scripts/eval_v21_rblend_scale.py \
  --additive-json "$CAL" \
  --dump-glob "$DUMPS/ablate_s2c_lt500_v22_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40 \
  --expected-tag lsst_r_extnbr_v22 --primary-domain rectangular \
  --re-min 0.5 --mag-max 25.8 --check-emulator-coverage \
  --output-json "$OUT"

echo V22_ANCHORBLEND_ADDITIVE_CONSTGOLD_DONE; date
