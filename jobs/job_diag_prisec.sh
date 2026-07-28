#!/bin/bash
#SBATCH --job-name=prisec
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/prisec_%j.out

# blendemu splits the input catalogue in half by index: first half = unsheared "primaries",
# second half = sheared "secondaries". The emulator trains on (first-half target, second-half
# neighbour); the half-shear ruler scores on second-half targets only (job 15328681). If the two
# halves are the same population the emulator transfers; if the catalogue order is sorted by any
# property, it does not -- and every cross-catalogue number in this investigation needs revisiting.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### PRIMARY/SECONDARY POPULATION job=$SLURM_JOB_ID ###"; date
python -u scripts/diag_primary_secondary.py --case ${CASE:-0} 2>&1 | grep -v "module command"
echo PRISEC_DONE; date
