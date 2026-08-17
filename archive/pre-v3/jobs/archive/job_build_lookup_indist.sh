#!/bin/bash
#SBATCH --job-name=lk_indist
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/lk_indist_%j.out

# Per-object summed R_blend on constgold from the INPUT-FRAME-distance emulator, so the certified m
# chain can be rerun with it. Mirrors jobs/job_build_4079.sh + job_build_100.sh exactly (same script,
# same cases 40-139, same --sign) with only --tag changed, so the lookup is a drop-in replacement for
# results/blend_lookup_extnbrho_c40-139.feather in validate_constant_with_blend.py.
#
# WHY (WORKLOG 2026-07-28n): the emulator was trained on detected-centroid separations but is queried
# here with input-catalogue separations (predict_response -> icat2reg -> make_reg_features), a
# train/inference mismatch worth 0.11" at sub-arcsecond separation. `lsst_r_extnbr_indist` is the same
# model retrained on the same labels with `distance` recomputed input-to-input.
#
# NOTE this reads constgold only to supply POSITIONS AND TRUE PROPERTIES for the lookup, exactly as
# the certified lookup build does; no constgold measurement enters the emulator or its training.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
mkdir -p results
echo "### blend_lookup 40-139, tag=${TAG:-lsst_r_extnbr_indist} job=$SLURM_JOB_ID ###"; date
python -u scripts/build_blend_lookup.py --cases $(seq 40 139) --tag ${TAG:-lsst_r_extnbr_indist} \
  --output results/blend_lookup_${SUFFIX:-indist}_c40-139.feather 2>&1 | grep -v "module command"
if [ -f results/blend_lookup_${SUFFIX:-indist}_c40-139.feather ]; then echo LK_INDIST_DONE;
else echo "LK_INDIST_FAILED (no output)"; exit 1; fi
date
