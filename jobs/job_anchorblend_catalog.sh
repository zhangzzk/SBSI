#!/bin/bash
#SBATCH --job-name=ab_cat
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_cat_%j.out
set -euo pipefail
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH="$SIMS/bin:$PATH"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
CFG=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_anchorblend_g005.yaml
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005
cd /home/z/Zekang.Zhang/blendemu/scripts
python -u run_pipeline.py --config "$CFG" --steps 1
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
python -u scripts/prepare_anchorblend_catalogues.py --base "$BASE" --cases $(seq 0 99) --g 0.05 --min-separation 20
echo ANCHORBLEND_CATALOG_DONE
