#!/bin/bash
#SBATCH --job-name=v36_proposal_coords
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/v36_proposal_coords_%j.out
#SBATCH --error=logs/v36_proposal_coords_%j.out
set -euo pipefail

# CPU-only diagnostic: re-rank the frozen worst-32 exact panel under
# alternative proposal coordinates.  No GPU is requested and no flow is
# evaluated -- this reuses the cached proposal coordinate table and the exact
# 24m posterior weights already on disk, so it does not touch the owner's
# two-GPU budget.  Proposal-only; the target, model, cuts and prior are fixed.

WORKTREE=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/proposal-coord-compare
RUN=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1

cd "$WORKTREE"
module load python/3.11-2023.09
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py31

export PYTHONPATH="$WORKTREE"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python -m pytest tests/test_proposal_coordinate_compare.py -q

python scripts/compare_proposal_coordinates.py \
  --run "$RUN" \
  --exact-dir "$RUN/tail_exact_16617850" \
  --retrieval "$RUN/proposal_probe_16617724/retrieval.json" \
  --output "$RUN/proposal_coordinate_compare_${SLURM_JOB_ID}"
