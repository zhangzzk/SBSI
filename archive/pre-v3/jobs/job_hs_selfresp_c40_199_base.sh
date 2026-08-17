#!/bin/bash
#SBATCH --job-name=hsb40199
#SBATCH --time=03:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsb40199_%j.out

# STAGE A of the FRESH half-shear self-response dump (cases 40-199).
#
# WHY A SEPARATE STAGE. The matched both-detected population build is pure CPU and IO-bound: it
# reads ~80% of two ~72 GB half-shear legs (~115 GB). Scoring is GPU and linear in checkpoints.
# Fanning 16 checkpoints out over a Slurm array without a cache would re-read those 115 GB sixteen
# times. This stage pays the read ONCE and writes the built population (with the per-object shear
# directions and gmed) to a cache; stage B's array tasks load the cache and go straight to the GPU.
#
# --case-chunk 80 builds [40,119] then [120,199] and concatenates. Every cut is per-object and the
# two legs are matched WITHIN a case, so this is exact; the only non-per-object quantity, gmed, is
# ASSERTED identical across chunks by the script and the run aborts if it is not.
#
# No GPU is requested: --build-base-only returns before any checkpoint is touched.
#
# FRESH-DATA BLIND: cases 40-199 have never been examined for the response-modulation question.
# Nothing here computes or prints a response.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
python -c "import torch,sys; sys.exit(0 if torch.__version__>='2' else 1)" \
  || { echo "WRONG PYTHON/TORCH -- conda activate sims1 did not take"; exit 1; }

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_c40-199/base_c40-199.feather
mkdir -p $(dirname $CACHE)

echo "### HS SELFRESP BASE CACHE cases 40-199  job=$SLURM_JOB_ID ###"; date
python -u scripts/dump_halfshear_selfresp.py \
  --min-case 40 --max-case 199 --case-chunk 80 \
  --base-cache $CACHE --build-base-only --blind \
  2>&1 | grep -v --line-buffered "module command" || { echo HSB_FAILED; exit 1; }
ls -la $CACHE $(dirname $CACHE)/base_c40-199.json
echo HSB_DONE; date
