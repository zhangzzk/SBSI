#!/bin/bash
#SBATCH --job-name=v36_atom_map
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=logs/v36_atom_map_%j.out
#SBATCH --error=logs/v36_atom_map_%j.out
set -euo pipefail

# CPU-only diagnostic figure: where the prior atoms sit in measured space and
# in truth space for one observation, which ones the production proposal drew,
# and which ones actually carry the posterior.  No GPU is requested and no flow
# is evaluated -- this reuses the cached proposal coordinate table, the cached
# detection probabilities and the exact 24m posterior weights already on disk,
# so it does not touch the owner's two-GPU budget.  Diagnostic only; the
# target, model, cuts and prior are untouched.

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
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl_${SLURM_JOB_ID}"

python -m pytest tests/test_proposal_atom_map.py -q

for ROW in 142230 3563 409188; do
  python scripts/plot_proposal_atom_map.py \
    --run "$RUN" \
    --exact-dir "$RUN/tail_exact_16617850" \
    --row "$ROW" \
    --output "$RUN/proposal_atom_map_${SLURM_JOB_ID}/row_${ROW}"
done
