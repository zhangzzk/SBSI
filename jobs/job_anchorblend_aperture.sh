#!/bin/bash
#SBATCH --job-name=ab_aperture
#SBATCH --time=04:00:00
#SBATCH --mem=240G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_aperture_%j.out
set -euo pipefail
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH="$SIMS/bin:$PATH"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
test ! -e results/anchorblend_g005_aperture_300.json
python -u scripts/eval_anchorblend_aperture.py \
  --response results/anchorblend_g005_response.feather results/anchorblend_g005_response_ext.feather \
  --base-old /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005 \
  --base-new /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext \
  --output-json results/anchorblend_g005_aperture_300.json
echo ANCHORBLEND_APERTURE_JOB_DONE
