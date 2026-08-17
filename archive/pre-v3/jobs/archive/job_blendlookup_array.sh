#!/bin/bash
#SBATCH --job-name=blk_arr
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --array=0-7
#SBATCH --output=/home/z/Zekang.Zhang/logs/blk_arr_%A_%a.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
# 8 chunks x 5 cases = cases 0..39
START=$(( SLURM_ARRAY_TASK_ID * 5 ))
CASES=$(seq $START $(( START + 4 )))
echo "task $SLURM_ARRAY_TASK_ID -> cases: $CASES"
python -u scripts/build_blend_lookup.py --cases $CASES \
  --output results/blend_lookup_const_part${SLURM_ARRAY_TASK_ID}.feather 2>&1 | grep -v module
echo "BLK_ARR_DONE task $SLURM_ARRAY_TASK_ID"
