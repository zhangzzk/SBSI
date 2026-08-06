#!/bin/bash
#SBATCH --job-name=lk_v22
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/lk_v22_%j.out
set -euo pipefail

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
OUT=results/blend_lookup_v22_c40-139.feather
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
echo "### V2.2 CONSTGOLD TRUE-PROPERTY LOOKUP job=$SLURM_JOB_ID ###"; date
python -u scripts/build_blend_lookup.py --cases $(seq 40 139) --tag lsst_r_extnbr_v22 \
  --output "$OUT" 2>&1 | grep -v --line-buffered "module command"
[ -f "$OUT" ] || { echo "LK_V22_FAILED: no output"; exit 1; }
echo LK_V22_DONE; date
