#!/bin/bash
#SBATCH --job-name=v22rk_cat
#SBATCH --array=0-18%6
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22rk_cat_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22rk_cat_%A_%a.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BE=/home/z/Zekang.Zhang/blendemu
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CFG=$ROOT/configs/fs2_lsst_r_v22_matched_decomp_pilot.yaml
PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_v22decomp_rankpilot
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
MODE=rank${TASK}
BASE=${PREFIX}_${MODE}
if [ -e "$BASE" ]; then
  echo "REFUSING: rank-pilot base already exists: $BASE"
  exit 1
fi
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:$BE:${PYTHONPATH:-}"
cd "$BE/scripts"
"$PY" -u run_pipeline.py --config "$CFG" --steps 1 --output-path "$BASE"
echo "V22_MATCHED_RANKPILOT_CATALOG_DONE mode=$MODE"
