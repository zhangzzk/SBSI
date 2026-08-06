#!/bin/bash
#SBATCH --job-name=v21_abiso300
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/v21_abiso300_%j.out
set -euo pipefail
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH="$SIMS/bin:$PATH"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
python -u scripts/eval_v21_rblend_scale.py \
  --isotonic-npz results/anchorblend_g005_isotonic_300.npz \
  --isotonic-summary results/anchorblend_g005_isotonic_300.json \
  --dump-glob '/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v21_domain_dumps/ablate_s2c_lt500_v21_perobj_s*.feather' \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather \
  --min-case 40
