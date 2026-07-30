#!/bin/bash
#SBATCH --job-name=cg_meas
#SBATCH --time=02:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_meas_%j.out

# Rebuild constgold WITH per-leg measured mag + size, so the selection test can cut sim and model on
# the SAME real quantity instead of a mag+size PROXY for S/N. Drives blendemu (catalogue building
# belongs there); adds 4 columns, computes nothing new -- both are plain SExtractor outputs already
# in the shape catalogue the builder loads.
#
# WRITES TO A SEPARATE PATH. `run_pipeline.py step_constant_catalogs` would DELETE the certified
# response catalogue before rebuilding; constgold is evaluation-only and has a documented history of
# stale-vs-fixed version confusion, so nothing existing is touched. --verify-against then requires
# every pre-existing column to be reproduced exactly (values AND NaN placement) before the new file
# is usable; a single differing column exits non-zero.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
D=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
echo "### REBUILD CONSTGOLD +meas job=$SLURM_JOB_ID ###"; date
python -u /home/z/Zekang.Zhang/blendemu/scripts/rebuild_constant_response_meas.py \
  --out-prefix "$D/constant_response_catalogue_meas" \
  --n-cases 140 --n-jobs 16 \
  --verify-against "$D/constant_response_catalogue_train.feather" \
  --tmp-dir "$D/_rebuild_parts" \
  2>&1 | grep -v "module command" || { echo CG_MEAS_FAILED; exit 1; }
echo CG_MEAS_ALL_DONE; date
