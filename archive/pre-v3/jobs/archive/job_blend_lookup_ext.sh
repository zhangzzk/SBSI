#!/bin/bash
#SBATCH --job-name=blend_lk_ext
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --array=0-7
#SBATCH --output=/home/z/Zekang.Zhang/logs/blend_lk_ext_%A_%a.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
START=$(( 40 + SLURM_ARRAY_TASK_ID * 20 )); CASES=$(seq $START $((START+19)))
echo "r_blend cases: $CASES"
python -u scripts/build_blend_lookup.py --cases $CASES \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 --sign 0.0 \
  --output results/blend_ext_part${SLURM_ARRAY_TASK_ID}.feather 2>&1 | grep -v module
echo "BLEND_LK_EXT_DONE ${SLURM_ARRAY_TASK_ID}"
