#!/bin/bash
#SBATCH --job-name=pi_grid
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a40-16gb:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/pi_grid_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/pi_grid_%j.err
#
# Refine the POPULATION block's node bank while leaving the score pass (and therefore the
# 5.7 h score cache) untouched.  scripts/check_quadrature.py measured the Pi-weighted
# integral to be 2.4% off at the production grid_n=61 -- 0.90% on `m`, larger than the whole
# residual -- while the per-galaxy machinery is already good to 0.04%.  Since the two are
# different integrals, only the population one needs refining, and since the Pi fast path it
# costs minutes.  If d(m) walks toward zero along this ladder, closure is a quadrature
# artefact and this is the fix.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
case "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" in
  *[0-9]Q|*[0-9]A|*[0-9]B) unset PYTORCH_CUDA_ALLOC_CONF ;;
esac
SC=/project/ls-gruen/users/zekang.zhang/sbsi_scores
# The score cache is keyed by the injected shear, so CLOSURE_G selects BOTH the `--closure-g`
# passed to the estimator and the cache file read back.  They must agree: `eval_score_select`
# checks `closure_g` in the cache key and refuses a mismatch, but keeping one variable here
# means a g=0.10 ladder cannot silently re-report the g=0.05 numbers.
CG=${CLOSURE_G:-0.05}
GTAG="g$(printf '%02d' "$(python -c "print(round(float('$CG')*100))")")"
date; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
echo "closure_g=$CG  cache tag=$GTAG"
# `sbatch --export` splits its OWN argument on commas, so `--export=ALL,PGRIDS=61,101,141`
# silently delivers `PGRIDS=61` and drops the rest -- the job then runs one rung and exits 0,
# which reads as success.  Use COLONS when exporting; both separators are accepted here so an
# interactive `PGRIDS="61 101 141"` still works.
split_list() { echo "$1" | tr ':,' '  '; }
# AZIMS sweeps the --pi-azimuthal-average DIAGNOSTIC (0 = raw, 1 = averaged).  Sweeping it
# inside one job is what makes the comparison airtight: Pi is drawn from a fixed seed and the
# averaging is applied AFTER the draw, so the two rungs share an identical Pi array and differ
# ONLY by the angular collapse.  `AZIMS=0:1` therefore isolates the flow anisotropy exactly.
# An averaged Pi is NOT a calibration -- d(m) from it is a diagnostic number, never a result.
# `grep` block-buffers at 4 kB when its stdout is a file, so a job killed at its time limit
# loses whatever is still in the buffer -- which is how a 3 h merge left a log holding nothing
# but its two header lines.  `--line-buffered` costs nothing and makes partial ladders readable.
# CUTS holds |xhat| radii; BOUNDS holds --cut-bound specs, whose ':' separators mean they
# must be split on commas/spaces only.  CACHE names the banked score pass to sweep, and it
# has to be the one built with THIS cut -- eval_score_select refuses the mismatch, so the
# failure is loud, but naming it per cut is what keeps the sweep honest.
for CUT in $(split_list "${CUTS:-0.6 0.4}"); do
  TAG=$(echo "$CUT" | tr -d '.' | sed 's/^0//')
  for PG in $(split_list "${PGRIDS:-61 81 101 141 181}"); do
   for AZ in $(split_list "${AZIMS:-0}"); do
    [ "$AZ" = 1 ] && AZFLAG=--pi-azimuthal-average || AZFLAG=
    echo ""; echo "######## cut=$CUT  closure_g=$CG  pi_grid_n=$PG  azim=$AZ ########"
    python -u scripts/eval_score_select.py \
      --closure-g "$CG" --cut-abs-ehat "$CUT" \
      --load-oversample "${OVERSAMPLE:-7.0}" --max-rows "${ROWS:-2000000}" \
      --pi-rows "${PIROWS:-1048576}" --pi-samples 8 --pi-reps 6 \
      --ring rot90 --shape-reps 2 --jk-blocks 200 --uncut-control \
      --pi-grid-n "$PG" $AZFLAG \
      --load-scores "${CACHE:-$SC/c0${TAG}_${GTAG}_8M_G2765.npz}" 2>&1 \
      | grep --line-buffered -vE "module command|Pi rep "
   done
  done
done
# The same ladder for an output-column cut (V3 only).  BOUNDS entries are separated by
# spaces or commas; each is one NAME:LO:HI passed straight through.
for B in ${BOUNDS:-}; do
  for PG in $(split_list "${PGRIDS:-61 81 101 141 181}"); do
   for AZ in $(split_list "${AZIMS:-0}"); do
    [ "$AZ" = 1 ] && AZFLAG=--pi-azimuthal-average || AZFLAG=
    echo ""; echo "######## cut=$B  closure_g=$CG  pi_grid_n=$PG  azim=$AZ ########"
    python -u scripts/eval_score_select.py \
      --closure-g "$CG" --cut-abs-ehat "${ABSCUT:-none}" --cut-bound "$B" \
      --load-oversample "${OVERSAMPLE:-7.0}" --max-rows "${ROWS:-2000000}" \
      --pi-rows "${PIROWS:-1048576}" --pi-samples 8 --pi-reps 6 \
      --ring rot90 --shape-reps 2 --jk-blocks 200 --uncut-control \
      --pi-grid-n "$PG" $AZFLAG \
      --load-scores "${CACHE:?BOUNDS needs CACHE=<banked score pass for this cut>}" 2>&1 \
      | grep --line-buffered -vE "module command|Pi rep "
   done
  done
done
date; echo "### DONE ###"
