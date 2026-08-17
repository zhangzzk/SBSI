#!/bin/bash
#SBATCH --job-name=agg26_cip
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/agg26_cip_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
DUMPS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v26_scene_cip_domain_dumps
echo "### V2.6 16-SEED CONSTGOLD ACCEPTANCE ###"; date
"$PY" -u scripts/eval_v2_indomain_m.py \
  --dump-glob "$DUMPS/ablate_s2c_lt500_v26_scene_target_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40 --re-min 0.5 --mag-max 25.8 \
  --emulator-tag lsst_r_extnbr_v22 --check-emulator-coverage
echo AGG_V26_CIP_DONE; date
