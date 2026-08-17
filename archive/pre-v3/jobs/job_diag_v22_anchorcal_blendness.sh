#!/bin/bash
#SBATCH --job-name=v22_calblend
#SBATCH --time=03:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22_calblend_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
DUMPS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps
CAL=results/anchorblend_g005_global_v22_c0-299.json
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

SCALE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_scale"])' "$CAL")
[ -n "${SCALE:?}" ] || exit 1
"$PY" -u scripts/diag_v22_blendness.py \
  --dump-glob "$DUMPS/ablate_s2c_lt500_v22_perobj_s*.feather" \
  --expect-seeds 16 --catalogue "$CAT" --min-case 40 \
  --mag-max 25.8 --re-min 0.5 --rblend-scale "$SCALE"

echo V22_ANCHORCAL_BLENDNESS_DONE; date
