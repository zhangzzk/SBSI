#!/bin/bash
#SBATCH --job-name=2sims
#SBATCH --time=03:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/2sims_%j.out
# 28k's unclosed blocker, now the last surviving explanation for the target defect (B)=+3.21%: do the
# TARGET's catalogue and the RULER's catalogue agree about the response for the SAME objects? Joins on
# (case,input_index) so population and selection differences are removed by construction. The ruler
# catalogue is 132 GB, hence the case cap and the memory request. Half-shear only -- no constgold.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### TWO SIMS AGREE? job=$SLURM_JOB_ID ###"; date
python -u scripts/eval_two_sims_agree.py --max-case ${MAXCASE:-4} \
  2>&1 | grep -v --line-buffered "module command" || { echo FAILED; exit 1; }
echo TWOSIMS_DONE; date
