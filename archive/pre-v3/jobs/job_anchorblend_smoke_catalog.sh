#!/bin/bash
#SBATCH --job-name=absm_cat
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/absm_cat_%j.out
set -euo pipefail
SIMS=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH="$SIMS/bin:$PATH"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
CFG=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_anchorblend_smoke.yaml
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_smoke
cd /home/z/Zekang.Zhang/blendemu/scripts
python -u run_pipeline.py --config "$CFG" --steps 1
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
python -u scripts/prepare_anchorblend_catalogues.py --base "$BASE" --cases 0 1 --min-separation 20
echo ANCHORBLEND_SMOKE_CATALOG_DONE
