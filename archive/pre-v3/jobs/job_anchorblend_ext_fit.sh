#!/bin/bash
#SBATCH --job-name=abx_fit
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/abx_fit_%j.out
set -euo pipefail
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH="$SIMS/bin:$PATH"
export LD_LIBRARY_PATH="$SIMS/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
test ! -e results/anchorblend_g005_isotonic_300.npz
test ! -e results/anchorblend_g005_isotonic_300.json
python -u scripts/fit_anchorblend_isotonic.py \
  results/anchorblend_g005_response.feather \
  results/anchorblend_g005_response_ext.feather \
  --tag lsst_r_extnbr_v21 --split-case 100 \
  --output-npz results/anchorblend_g005_isotonic_300.npz \
  --output-json results/anchorblend_g005_isotonic_300.json
echo ANCHORBLEND_EXT_FIT_DONE
