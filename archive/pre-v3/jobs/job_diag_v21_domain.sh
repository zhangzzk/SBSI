#!/bin/bash
#SBATCH --job-name=diag_v21dom
#SBATCH --time=02:00:00
#SBATCH --mem=180G         # 16 dumps x 27M rows float32 = 1.7G, plus a 41.7M-row catalogue join
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag_v21dom_%j.out

# Scores the FIDUCIAL V2 model on the V2.1 population, to decide whether V2.1's +0.841% no-cut m is
# a property of the POPULATION or of the RETRAIN. See the script docstring for the full argument.
#
# CPU only: it reads existing per-object dumps and joins a catalogue column. No flow is evaluated,
# so there is nothing for a GPU to do.
#
# FIREWALL: read-and-print over existing products. Nothing trains, tunes or selects a model.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### DIAG V2.1 DOMAIN SUBSET job=$SLURM_JOB_ID ###"; date
python -u scripts/diag_v21_domain_subset.py 2>&1 | grep -v --line-buffered "module command" \
  || { echo DIAG_V21_FAILED; exit 1; }
echo DIAG_V21_ALL_DONE; date
