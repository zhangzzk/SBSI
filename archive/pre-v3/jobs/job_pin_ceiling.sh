#!/bin/bash
#SBATCH --job-name=pin_ceil
#SBATCH --time=04:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/pin_ceil_%j.out
set -o pipefail

# STEP 2 of the PIN RE-ALLOCATION test: build three grids, harvest each as a per-object R_flow
# oracle, and score all three plus the real fiducial flow on the same constgold rows.
#
# NO GPU. A trained flow can approach its response target but never beat it, so the target's own
# per-cell value IS that grid's ceiling. This prices the edge re-allocation before any retrain.
#
# FOUR ARMS, one variable at a time:
#   A  6x6  equal-count       -- the CURRENT allocation (quantile edges), the control
#   B  6x6  response-driven   -- equal d(log R) edges, CONSERVATIVE (--min-frac 0.04; the size axis
#                                retreated 40% toward equal-count to clear that floor)
#   B2 6x6  response-driven   -- same criterion, PURE (--min-frac 0.02, alpha = 0 on both axes).
#                                Still 2,354 per cell at 6x6x5, above the 1,875 floor that rejected
#                                the 8x8 grid. B vs B2 prices how much the statistical guard costs,
#                                so the guard is measured rather than assumed.
#   C 20x20 equal-count       -- a RESOLUTION upper bound. C answers the question A-vs-B cannot:
#                                if C is barely better than A, binning is not the binding constraint
#                                at all and per-object supervision is the only route left.
#
# ALL THREE ARE flux x size WITH ONE CROWD BIN. harvest_grid_perobj.py assigns the 3rd axis from
# distance/neighbored, but the fiducial grid's 3rd axis is r_blend, which the constgold catalogue
# does not carry. So these ceilings are NOT the fiducial 6x6x5 pin's ceiling -- they are a
# controlled A/B/C at a common axis structure. The comparison is the result; the absolute value is
# a floor on the fiducial grid, not an estimate of it. This is stated again in the eval output.
#
# FIREWALL: every grid is built from the half-shear g=0.05 leg + its g=0 SNC lookup. constgold
# supplies lookup COORDINATES and the scoring r_sim only. The edges were fixed on the ruler before
# this job ran; no grid is selected on an m.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
OUTD=/project/ls-gruen/users/zekang.zhang/sbsi_caches/pin_realloc
SNC=/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather
CONSTCAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
mkdir -p "$OUTD"

# Edges from scripts/make_response_edges.py on the 60-bin ruler profiles (job 15489742), fixed
# BEFORE this job runs and with no m in evidence. The flux axis is identical for B and B2 because
# it already placed at alpha = 0 under the conservative floor; only the size axis differs.
FLUX_EDGES=${FLUX_EDGES:-18.002082,24.115564,24.790108,25.175071,25.506239,25.795854,25.999993}
SIZE_EDGES=${SIZE_EDGES:-0.299999,0.316804,0.330889,0.358446,0.420956,0.618024,1.500001}
SIZE_EDGES2=${SIZE_EDGES2:-0.299999,0.313488,0.320842,0.331555,0.349768,0.390145,1.500001}

echo "### PIN CEILING job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
echo "FLUX_EDGES  = $FLUX_EDGES"
echo "SIZE_EDGES  = $SIZE_EDGES   (B,  min-frac 0.04, alpha 0.40)"
echo "SIZE_EDGES2 = $SIZE_EDGES2   (B2, min-frac 0.02, alpha 0.00)"

build () {   # $1=output npz, rest = grid spec
  local OUT="$1"; shift
  python -u scripts/compute_response_target_blend.py \
    --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
    --target-cols measured_ngmix_g1 measured_ngmix_g2 \
    --nominal-g 0.05 --crowd-col r_blend --n-crowd 1 \
    --min-count 500 --max-case 99 \
    --primary-mag-max 26.0 --primary-re-min 0.3 \
    --snc-lookup "$SNC" --output "$OUT" "$@" \
    2>&1 | grep -v --line-buffered "module command" || exit 1
}

harvest () {  # $1=grid npz  $2=override npz
  python -B -u scripts/harvest_grid_perobj.py \
    --grid "$1" --catalogue $CONSTCAT --min-case 40 --out "$2" \
    2>&1 | grep -v --line-buffered "module command" || exit 1
}

echo; echo "########## A: 6x6 EQUAL-COUNT (control) ##########"
build $OUTD/grid_A_eqcount_6x6.npz --n-flux 6 --n-size 6
harvest $OUTD/grid_A_eqcount_6x6.npz $OUTD/oracle_A_eqcount_6x6.npz

echo; echo "########## B: 6x6 RESPONSE-DRIVEN (conservative) ##########"
build $OUTD/grid_B_respdriven_6x6.npz --flux-edges "$FLUX_EDGES" --size-edges "$SIZE_EDGES"
harvest $OUTD/grid_B_respdriven_6x6.npz $OUTD/oracle_B_respdriven_6x6.npz

echo; echo "########## B2: 6x6 RESPONSE-DRIVEN (pure, alpha=0) ##########"
build $OUTD/grid_B2_respdriven_6x6.npz --flux-edges "$FLUX_EDGES" --size-edges "$SIZE_EDGES2"
harvest $OUTD/grid_B2_respdriven_6x6.npz $OUTD/oracle_B2_respdriven_6x6.npz

echo; echo "########## C: 20x20 EQUAL-COUNT (resolution upper bound) ##########"
build $OUTD/grid_C_eqcount_20x20.npz --n-flux 20 --n-size 20
harvest $OUTD/grid_C_eqcount_20x20.npz $OUTD/oracle_C_eqcount_20x20.npz

echo; echo "########## CEILING SCORES ##########"
python -u scripts/eval_pin_ceiling.py \
  --oracle "A eq-count 6x6=$OUTD/oracle_A_eqcount_6x6.npz" \
  --oracle "B resp-driven 6x6=$OUTD/oracle_B_respdriven_6x6.npz" \
  --oracle "B2 resp-pure 6x6=$OUTD/oracle_B2_respdriven_6x6.npz" \
  --oracle "C eq-count 20x20=$OUTD/oracle_C_eqcount_20x20.npz" \
  2>&1 | grep -v --line-buffered "module command" || exit 1

echo; echo "PIN_CEIL_DONE"; date
