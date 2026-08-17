#!/bin/bash
#SBATCH --job-name=v26blendcmp
#SBATCH --time=01:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v26blendcmp_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
V22=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps
V26=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v26_scene_cip_domain_dumps
"$PY" -u scripts/diag_v22_blendness.py \
  --dump-glob "$V26/ablate_s2c_lt500_v26_scene_target_perobj_s*.feather" \
  --reference-dump-glob "$V22/ablate_s2c_lt500_v22_perobj_s*.feather" \
  --expect-seeds 6 --catalogue "$CAT" --min-case 40 --mag-max 25.8 --re-min 0.5
echo V26_V22_BLENDNESS_COMPARE_DONE; date
