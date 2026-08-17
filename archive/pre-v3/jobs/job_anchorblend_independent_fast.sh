#!/bin/bash
#SBATCH --job-name=ab_indfast
#SBATCH --time=02:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --array=0-19%20
#SBATCH --exclude=usm-cl-183r01,usm-cl-826bac01,usm-cl-826bac02,usm-cl-826bac03,usm-cl-1116cs01,usm-cl-1116cs02,usm-cl-1116cs03,usm-cl-seitz3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_indfast_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_indfast_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_independent_local10_c200-299
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_independent_fast_c200-299
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:$ROOT/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$PARTS"
cd "$ROOT"
START=$((200 + 5 * SLURM_ARRAY_TASK_ID))
END=$((START + 4))
for case in $(seq "$START" "$END"); do
  env -u SLURM_PROCID python -u scripts/measure_anchor_independent_fast.py \
    --base "$BASE" --output-dir "$PARTS" \
    --case-offset "$case" --n-cases 1
done
echo "ANCHORBLEND_INDEPENDENT_FAST_PART_DONE cases=$START-$END"
date
