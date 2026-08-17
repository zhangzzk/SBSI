#!/bin/bash
#SBATCH --job-name=v22_abiso
#SBATCH --time=03:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22_abiso_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
DUMPS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps
OUT=results/v22_anchorblend_isotonic_constgold_c40-139_s16.json
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }
"$PY" -u scripts/eval_v21_rblend_scale.py \
  --isotonic-npz results/anchorblend_g005_isotonic_v22_c0-299.npz \
  --isotonic-summary results/anchorblend_g005_isotonic_v22_c0-299.json \
  --dump-glob "$DUMPS/ablate_s2c_lt500_v22_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40 \
  --expected-tag lsst_r_extnbr_v22 --primary-domain rectangular \
  --re-min 0.5 --mag-max 25.8 --check-emulator-coverage \
  --output-json "$OUT"

echo V22_ANCHORBLEND_ISOTONIC_CONSTGOLD_DONE; date
