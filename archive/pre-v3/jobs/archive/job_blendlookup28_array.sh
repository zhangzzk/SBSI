#!/bin/bash
#SBATCH --job-name=blk28_arr
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --array=0-7
#SBATCH --output=/home/z/Zekang.Zhang/logs/blk28_arr_%A_%a.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
START=$(( SLURM_ARRAY_TASK_ID * 5 ))
CASES=$(seq $START $(( START + 4 )))
echo "task $SLURM_ARRAY_TASK_ID -> cases: $CASES  (r<28, r_max=10, k=20)"
python -u scripts/build_blend_lookup.py --cases $CASES \
  --output results/blend_lookup_const28_part${SLURM_ARRAY_TASK_ID}.feather 2>&1 | grep -v module
echo "BLK28_ARR_DONE task $SLURM_ARRAY_TASK_ID"
