#!/bin/bash
# Regenerate the proposal atom map for the worst-curvature rows.
#
# Diagnostic only: nothing here touches the target, model, cuts, prior or any
# production setting.  It reproduces the sampler's draw for one observation and
# plots where the atoms that actually carry posterior mass end up.
#
#   sbatch scripts/run_proposal_atom_map.sh [production|fifty-fifty]
#
# "production" races one mixture and lets the ranked/flat mass ratio decide how
# the 16,384 slots are split.  "fifty-fifty" gives each stratum half the slots
# outright.  No GPU: the proposal coordinates and detection probabilities are
# read from the prepared cache.
#SBATCH --job-name=v36_atom_map
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v36_atom_map_%j.out
set -euo pipefail

SAMPLER="${1:-production}"
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
RUN=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_v36_uncut_size060_n500k_m16384_20260920_v1
EXACT="$RUN/tail_exact_16617850"
OUT="$RUN/proposal_atom_map_${SLURM_JOB_ID}"

cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$PY" -m pytest tests/test_proposal_atom_map.py -q

for ROW in 142230 3563 409188; do
  "$PY" scripts/plot_proposal_atom_map.py \
    --run "$RUN" \
    --exact-dir "$EXACT" \
    --row "$ROW" \
    --sampler "$SAMPLER" \
    --output "$OUT/row_${ROW}"
done
