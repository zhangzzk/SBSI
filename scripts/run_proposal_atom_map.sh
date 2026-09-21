#!/bin/bash
# Regenerate the proposal atom map for the worst-curvature rows.
#
# Diagnostic only: nothing here touches the target, model, cuts, prior or any
# production setting.  It reproduces the configured `tilted_stratified` draw for
# one observation -- inverse-CDF from the defensive mixture, with replacement,
# with the exactly summed stratum left in the mixture so draws landing there are
# spent but carry nothing -- and plots where the atoms that actually carry
# posterior mass end up.
#
#   sbatch scripts/run_proposal_atom_map.sh [FLOOR_PERCENTILE] [FRACTIONAL_TARGET...]
#
# With no arguments the proposal's predicted scatter is used exactly as the
# cached table ships it, floored at the 1st percentile.  Passing a percentile
# re-floors that scatter before the mixture is built, which is the one knob that
# decides how tightly a vague atom may be judged; any further arguments name
# coordinates whose scatter is multiplicative, so their floor is taken on
# sigma/|x| and binds at every brightness.  Flux needs this: it spans five
# decades, so an absolute floor is fixed by the faintest atoms and can never
# bind on a bright one.  No GPU: the proposal coordinates and detection
# probabilities are read from the prepared cache.
#
# Example, the setting measured to reach 90.6% of the captured mass against
# 56.9% as shipped:
#
#   sbatch scripts/run_proposal_atom_map.sh 50 measured_flux_from_mag_auto
#
#SBATCH --job-name=v36_atom_map
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v36_atom_map_%j.out
set -euo pipefail

FLOOR="${1:-}"
shift || true
FLOOR_ARGS=()
if [[ -n "$FLOOR" ]]; then
  FLOOR_ARGS+=(--floor-percentile "$FLOOR")
  for TARGET in "$@"; do
    FLOOR_ARGS+=(--fractional-floor "$TARGET")
  done
fi

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
    ${FLOOR_ARGS[@]+"${FLOOR_ARGS[@]}"} \
    --output "$OUT/row_${ROW}"
done
