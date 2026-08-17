#!/bin/bash
#SBATCH --job-name=ra_target
#SBATCH --time=04:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/ra_target_%j.out

# STAGE P1-C: build the REALISATION-MODULATION target rho(cell, measured sub-bin) for the RA head.
#
# TWO STEPS, both here:
#   (1) EXTEND the g=0 SNC lookup with the g0 leg's MEASURED mag and flux radius. The existing
#       results/g0_lookup_c0-99.feather carries only ngmix0_g1/g2, and the sub-bin label has to be
#       the UNSHEARED leg's measured photometry. build_g0_lookup.py gained --extra-cols for this;
#       with no --extra-cols its output is byte-identical to the historical lookup. NEW PATH, so
#       the certified lookup is untouched.
#   (2) BUILD rho on the same half-shear g=0.05 leg and the same population cuts the certified
#       per-cell target used, with the coarse cells a strict MERGE of that target's 6x6x5 grid.
#
# CASE RANGE. The sheared catalogue det_meas_crowd_g0.05_val_full.feather covers cases 0-99 ONLY
# (verified: last record batch is case 99), so the fit/held-out split lives inside 0-99: fit 0-79,
# held out 80-99. The design's "100-139 held out" is NOT available from this catalogue; extending it
# would need a g=0.05 render of those cases, not just a bigger lookup.
#
# Also emits the two control targets -- the 45-degree null and the sheared-leg-binned variant --
# which are diagnostics only and must never be trained on.
#
# THE ONE REAL FREE PARAMETER IS THE CELL GRANULARITY, and the smoke run (job 15423571) showed why
# it matters: at merge 3/3/5 (only 4 coarse cells) the FIDUCIAL, realisation-BLIND model already
# reproduced rho closely (rho_model 1.44/0.70 against rho_sim 1.48/0.65 at the extreme sub-bins).
# That is not the model being realisation-aware -- it is the coarse cell letting the measured
# sub-bin proxy for TRUE properties the model does see. The coarser the cell, the more of rho is
# already-satisfied true-property structure and the smaller the genuinely learnable residual.
# So both granularities are built and the DEFAULT is the finer one (merge 2/2/1 -> 3x3x5 = 45
# cells). Choose between them on the P0 probe / held-out rho closure -- both half-shear, both
# firewall-clean -- NEVER on a constgold number.
#
# FIREWALL: half-shear legs + their g=0 lookup. No constgold, no emulator, no m.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
R=/home/z/Zekang.Zhang/SBSI/results
OUTD=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ra
mkdir -p "$OUTD"
LOOKUP=$OUTD/g0_lookup_meas_c0-99.feather
FINE=$R/response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz
CAT=$D/det_meas_crowd_g0.05_val_full.feather

echo "### RA TARGET job=$SLURM_JOB_ID ###"; date

# ---- (1) extended g0 lookup (skip if already built) -----------------------------------------
if [ -f "$LOOKUP" ]; then
  echo "reusing existing $LOOKUP"
else
  python -u scripts/build_g0_lookup.py \
    --cases $(seq 0 99) --output "$LOOKUP" \
    --extra-cols MAG_AUTO:measured_mag_auto_g0 FLUX_RADIUS:measured_flux_radius_g0 \
    2>&1 | tail -20 || { echo LOOKUP_FAILED; exit 1; }
fi

# ---- (2) rho targets -------------------------------------------------------------------------
build () {   # $1=label  $2=outfile  $3.. = extra flags
  local L="$1"; local O="$2"; shift 2
  echo; echo "--- $L -> $(basename "$O") ---"
  python -u scripts/compute_response_target_measbin.py \
    --catalogue "$CAT" --fine-target-npz "$FINE" --snc-lookup "$LOOKUP" \
    --target-cols measured_ngmix_g1 measured_ngmix_g2 --nominal-g 0.05 \
    --merge-flux ${MF:-2} --merge-size ${MS:-2} --merge-crowd ${MC:-1} \
    --n-dmag ${NDMAG:-5} --n-dlogsize ${NDLSZ:-2} --min-count ${MINCNT:-500} \
    --primary-mag-max 26.0 --primary-re-min 0.3 \
    --output "$O" "$@" 2>&1 | grep -v --line-buffered "module command" \
    || { echo "FAILED $L"; exit 1; }
}

build "FIT (cases 0-79)"       "$OUTD/ra_target_g0meas_c0-79.npz"   --max-case 79
build "HELD OUT (cases 80-99)" "$OUTD/ra_target_g0meas_c80-99.npz"  --min-case 80 --max-case 99
build "NULL control (45 deg)"  "$OUTD/ra_target_null_c0-79.npz"     --max-case 79 --null-rotate
build "sheared-leg control"    "$OUTD/ra_target_gsmeas_c0-79.npz"   --max-case 79 --bin-on gsmeas
# coarse-cell alternative (4 cells) -- the granularity comparison, see the header note.
# The trailing --merge-* repeat the option, and argparse keeps the LAST occurrence, so these
# override the defaults inside build() without relying on env-prefix scoping around a function.
build "COARSE cells (merge 3/3/5)" "$OUTD/ra_target_g0meas_coarse_c0-79.npz" --max-case 79 \
      --merge-flux 3 --merge-size 3 --merge-crowd 5

echo RA_TARGET_DONE; date
