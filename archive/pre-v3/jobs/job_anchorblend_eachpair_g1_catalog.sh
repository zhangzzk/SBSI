#!/bin/bash
#SBATCH --job-name=abep_cat
#SBATCH --array=0-18%8
#SBATCH --time=02:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abep_cat_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abep_cat_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BE=/home/z/Zekang.Zhang/blendemu
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
CFG=$ROOT/configs/fs2_lsst_r_anchorblend_eachpair_g1_pilot_c400-409.yaml
PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_eachpair_g1
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
BASE=$(printf '%s_rank%02d_pilot_c400-409' "$PREFIX" "$TASK")
test ! -e "$BASE" || { echo "REFUSING existing $BASE"; exit 1; }
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:$BE:${PYTHONPATH:-}"
cd "$BE/scripts"
python -u run_pipeline.py --config "$CFG" --steps 1 --output-path "$BASE"
for case in $(seq 400 409); do
  test -s "$BASE/gals${case}_0.05.feather"
  test -s "$BASE/gals${case}_-0.05.feather"
done
echo "ANCHORBLEND_EACHPAIR_G1_CATALOG_DONE rank=$TASK"
date
