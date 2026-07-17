#!/bin/bash
#SBATCH --job-name=blk_extdom
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --array=0-7
#SBATCH --output=/home/z/Zekang.Zhang/logs/blk_extdom_%A_%a.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
M=/home/z/Zekang.Zhang/blendemu/models
[ -f $M/classification_model_lsst_r_extdom.json ] || cp $M/classification_model_lsst_r.json $M/classification_model_lsst_r_extdom.json
START=$(( SLURM_ARRAY_TASK_ID * 5 )); CASES=$(seq $START $((START+4)))
echo "extdom blend lookup, cases: $CASES"
python -u scripts/build_blend_lookup.py --cases $CASES --tag lsst_r_extdom \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant --sign 0.02 \
  --output results/blend_extdom_part${SLURM_ARRAY_TASK_ID}.feather 2>&1 | grep -v module
echo "BLK_EXTDOM_DONE ${SLURM_ARRAY_TASK_ID}"
