#!/bin/bash
#SBATCH --job-name=ab1n_cat
#SBATCH --array=0-1%2
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab1n_cat_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab1n_cat_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BE=/home/z/Zekang.Zhang/blendemu
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
CFG=$ROOT/configs/fs2_lsst_r_anchorblend_oneactive_orthogonal_c400-499.yaml
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
if [ "$TASK" -eq 0 ]; then MODE=u; else MODE=v; fi
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_oneactive_${MODE}_c400-499
test ! -e "$BASE" || { echo "REFUSING existing $BASE"; exit 1; }
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:$BE:${PYTHONPATH:-}"
cd "$BE/scripts"
python -u run_pipeline.py --config "$CFG" --steps 1 --output-path "$BASE"
echo "ANCHORBLEND_ONEACTIVE_CATALOG_DONE mode=$MODE"
date
