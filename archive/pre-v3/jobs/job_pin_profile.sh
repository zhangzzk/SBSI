#!/bin/bash
#SBATCH --job-name=pin_prof
#SBATCH --time=03:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/pin_prof_%j.out
set -o pipefail

# STEP 1 of the PIN RE-ALLOCATION ceiling test: FINE 1-D response profiles on true mag and true size.
#
# These are the input to scripts/make_response_edges.py, which places grid edges on equal relative
# response change instead of equal count. They are built by the CERTIFIED builder with the fiducial
# target's own catalogue, estimator, SNC lookup, nominal g and domain cuts -- only the bin counts
# differ -- so the profile is the fiducial pin measured at 60x resolution on one axis at a time, not
# a new quantity.
#
# Each run sets the other two axes to ONE bin, which makes the output a genuine marginal (the
# collapsed axes are count-weighted, exactly the weighting the marginal edges need). --n-crowd 1
# also keeps the 3rd axis a single bin so the npz stays a clean 1-D profile.
#
# FIREWALL: half-shear g=0.05 leg + its g=0 SNC lookup. constgold is not read; no m exists at this
# stage, and the edges are fixed before any acceptance number is computed.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
OUTD=/project/ls-gruen/users/zekang.zhang/sbsi_caches/pin_realloc
mkdir -p "$OUTD"
SNC=/home/z/Zekang.Zhang/SBSI/results/g0_lookup_c0-99.feather

echo "### PIN PROFILE job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date

common () {
  python -u scripts/compute_response_target_blend.py \
    --catalogue $D/det_meas_crowd_g0.05_val_full.feather \
    --target-cols measured_ngmix_g1 measured_ngmix_g2 \
    --nominal-g 0.05 --crowd-col r_blend --n-crowd 1 \
    --min-count 500 --max-case 99 \
    --primary-mag-max 26.0 --primary-re-min 0.3 \
    --snc-lookup "$SNC" "$@" 2>&1 | grep -v --line-buffered "module command"
}

echo; echo "########## R vs TRUE MAG, 60 equal-count bins ##########"
common --n-flux 60 --n-size 1 --output $OUTD/pin_profile_mag60.npz || exit 1

echo; echo "########## R vs TRUE SIZE, 60 equal-count bins ##########"
common --n-flux 1 --n-size 60 --output $OUTD/pin_profile_size60.npz || exit 1

echo; echo "PIN_PROF_DONE $OUTD"; date
