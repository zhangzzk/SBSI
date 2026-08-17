#!/bin/bash
#SBATCH --job-name=diagjoin
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/diagjoin_%j.out

# eval_shear_amplitude matched ZERO pairs between the response catalogue and the half-shear
# catalogue. Walk the key hierarchy (case -> input_index -> primary position -> neighbour position)
# and find where the agreement stops. See scripts/diag_catalogue_join.py.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### DIAG JOIN job=$SLURM_JOB_ID ###"; date
python -u scripts/diag_catalogue_join.py --case ${CASE:-0} ${TRACE:+--trace-load-legs} 2>&1 | grep -v "module command"
echo DIAGJOIN_DONE; date
