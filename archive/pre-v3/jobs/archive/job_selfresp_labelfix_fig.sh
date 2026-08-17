#!/bin/bash
#SBATCH --job-name=fig5_lblfix
#SBATCH --time=00:40:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/fig5_lblfix_%j.out
set -o pipefail

# Figure-5 axes with BOTH flows on them: the fiducial (ruler-label) flow and the label-consistent
# retrain (scored-label), on identical rows against the same half-shear truth. CPU only -- both dumps
# already exist, this just bins and bootstraps them.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### FIG5 LABELFIX job=$SLURM_JOB_ID ###"; date
python -u plotting/plot_selfresp_labelfix.py \
  2>&1 | grep -v --line-buffered "module command" || { echo FIG5_LBLFIX_FAILED; exit 1; }
echo FIG5_LBLFIX_DONE; date
