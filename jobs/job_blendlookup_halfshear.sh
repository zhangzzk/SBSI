#!/bin/bash
#SBATCH --job-name=blk_hs
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --array=0-7
#SBATCH --output=/home/z/Zekang.Zhang/logs/blk_hs_%A_%a.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876
START=$(( SLURM_ARRAY_TASK_ID * 5 ))
CASES=$(seq $START $(( START + 4 )))
echo "task $SLURM_ARRAY_TASK_ID -> half-shear cases: $CASES  (base=$BASE, sign=0.05, r<28/10in/k20)"
python -u scripts/build_blend_lookup.py --cases $CASES --base "$BASE" --sign 0.05 \
  --output results/blend_lookup_hs_part${SLURM_ARRAY_TASK_ID}.feather 2>&1 | grep -v module
echo "BLK_HS_DONE task $SLURM_ARRAY_TASK_ID"
