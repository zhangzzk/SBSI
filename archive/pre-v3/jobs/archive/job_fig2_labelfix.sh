#!/bin/bash
#SBATCH --job-name=fig2_lblfix
#SBATCH --time=01:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/fig2_lblfix_%j.out
set -o pipefail

# Figure-2 axes with BOTH flows on them: full model (flow + TUNED emulator R_blend) vs the constgold
# total response, fiducial vs the label-consistent retrain, identical rows and seeds. CPU only --
# both constgold dump sets already exist; this joins the tuned lookup, bins and bootstraps.
# Memory follows the constgold dump jobs: the per-object dumps are ~1 GB each and the property merge
# is on 27M rows.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### FIG2 LABELFIX job=$SLURM_JOB_ID ###"; date
python -u plotting/plot_fig2_labelfix.py \
  2>&1 | grep -v --line-buffered "module command" || { echo FIG2_LBLFIX_FAILED; exit 1; }
echo FIG2_LBLFIX_DONE; date
