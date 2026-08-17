#!/bin/bash
#SBATCH --job-name=hsm40199
#SBATCH --time=01:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsm40199_%j.out

# STAGE C: merge the 16 per-checkpoint parts into the fresh-case dump.
#
# Submit after the array:
#   sbatch --dependency=afterok:<ARRAY_JOBID> jobs/job_hs_selfresp_c40_199_merge.sh
#
# The merge refuses to combine parts whose (case, input_index) do not line up elementwise (falling
# back to an explicit 1:1 join before it gives up), whose independently recomputed sim columns
# disagree, whose seed set is not exactly the 16 dom6x6 e-response seeds, or whose case range
# reaches into the burned cases 0-39.
#
# BLIND: no response value is printed anywhere in this stage.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_c40-199
echo "### HS SELFRESP c40-199 MERGE job=$SLURM_JOB_ID ###"; date
python -u scripts/merge_halfshear_selfresp_parts.py \
  --parts "$OUTDIR/parts/part_s*.feather" \
  --out $OUTDIR/halfshear_selfresp_c40-199.feather \
  --min-case 40 --max-case 199 \
  2>&1 | grep -v --line-buffered "module command" || { echo HSM_FAILED; exit 1; }
ls -la $OUTDIR
echo HSM_DONE; date
