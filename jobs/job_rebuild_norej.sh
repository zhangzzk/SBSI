#!/bin/bash
#SBATCH --job-name=norejcat
#SBATCH --time=08:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/norejcat_%j.out

# Rebuild blendemu's response catalogue with the bright-neighbour rejection DISABLED, so the
# emulator's training labels cover the same population the m pipeline sums over. Pure catalogue
# build: the shape and cross-match catalogues already exist for every case, nothing is re-simulated.
# See scripts/rebuild_response_norej.py for why the flag is monkeypatched rather than edited into
# blendemu (that checkout has uncommitted work on main).
# Writes response_catalogue_${SUFFIX}_train.feather; refuses to overwrite an existing file.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### REBUILD RESPONSE (no bright-neighbour rejection) job=$SLURM_JOB_ID ###"; date
python -u scripts/rebuild_response_norej.py \
    --ratio-max ${RATIO_MAX:-inf} --suffix ${SUFFIX:-norej} \
    --n-jobs ${SLURM_CPUS_PER_TASK:-16} 2>&1 | grep -v "module command"
echo NOREJCAT_DONE; date
