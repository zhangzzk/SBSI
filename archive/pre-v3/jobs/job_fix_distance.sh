#!/bin/bash
#SBATCH --job-name=fixdist
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/fixdist_%j.out

# Measure (and optionally repair) the distance-definition mismatch INSIDE the emulator's training
# catalogue: its `distance` is detected-centroid-to-input-neighbour, while the half-shear catalogue
# the ruler and the m pipeline use is input-to-input. The response catalogue stores both input
# positions, so the input-frame separation is recomputable from the existing rows -- no
# re-simulation, no rebuild, labels untouched.
#   STATS_ONLY=1  -> measure only (fast, writes nothing)
#   STATS_ONLY=0  -> also write the corrected catalogue for retraining
# See scripts/fix_response_distance.py.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
echo "### FIX DISTANCE (stats_only=${STATS_ONLY:-1}) job=$SLURM_JOB_ID ###"; date
if [ "${STATS_ONLY:-1}" = "1" ]; then EXTRA="--stats-only"; else EXTRA=""; fi
python -u scripts/fix_response_distance.py $EXTRA 2>&1 | grep -v "module command"
echo FIXDIST_DONE; date
