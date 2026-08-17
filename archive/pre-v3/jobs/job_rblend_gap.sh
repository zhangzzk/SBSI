#!/bin/bash
#SBATCH --job-name=rblendgap
#SBATCH --time=01:00:00
#SBATCH --mem=64G          # MEASURED peak 41.9G (job 15477392). The old 200G was ~5x that; an
                           # identical over-request on the constgold table pended 14h behind free
                           # nodes. The domain cut lands AFTER the leg load, so V2.1 peaks the same.
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/rblendgap_%j.out

# Score the BlendEMU per-pair blending response against half-shear truth ON THE DELIVERABLE DOMAIN.
# R_flow is cleared (-0.05%, WORKLOG 2026-07-28h), so the in-domain m of -0.508% must live in R_blend
# or the population transfer. The emulator's training cuts are a SUPERSET of our domain (primary mag
# 18-28 vs <26, Re 0.1-1.5 vs >0.3) and its accuracy INSIDE the domain has never been measured.
#
# Truth = the NEIGHBOUR-ONLY-SHEARED leg of the half-shear 2x2 design. CPU-only (XGBoost on cpu).
# FIREWALL: no constgold is read.
#
# V21=1 switches the population to the V2.1 domain (true Re > 0.5" AND true S/N > 10, the curve from
# sbs_shear.domain) instead of the REMIN/MAGMAX rectangle, and re-spends the size-table edges above
# 0.5" so the axis does not collapse to two bins. Pair it with TAG=lsst_r_extnbr_v21 to score the
# V2.1 emulator where the V2.1 flow actually lives -- the fiducial-domain run (job 15477392, tag
# lsst_r_extnbr_indom_tuned) measured -13.12% there, which is the WRONG domain AND the wrong
# emulator for judging V2.1.
#
# THIS IS THE ONLY LEGITIMATE PLACE TO ARGUE ABOUT THE EMULATOR (AGENTS.md): promotion is decided on
# this per-pair ruler, NEVER on constgold m. A number from here may motivate an emulator change; a
# constgold m may not.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/eval; mkdir -p $D
TAG=${TAG:-lsst_r_extnbr_ho}
if [ "${V21:-0}" = "1" ]; then DOM="--v21-domain"; SUF="${OUTSUF:-_v21dom}"; else DOM=""; SUF="${OUTSUF:-}"; fi

# LEG OVERRIDES. Defaults are the script's own (non-ap7 g0.05 val vs g0.0 train). Set GS_LEG/G0_LEG
# to raise precision off the g=0.2 leg, which carries 4x the shear signal at similar shape noise.
# BOTH LEGS MUST COME FROM THE SAME APERTURE FAMILY, and a cross-family comparison is invalid: the
# ap7 catalogues are 7"-capped, and AGENTS.md's pair-list rule says restricting the pair list
# handicaps whichever model relies on the excluded pairs -- so an ap7 number may only be read
# against an ap7 control, never against the non-ap7 default above. Run ap7 g=0.05 as that control.
LEGS=""
[ -n "$GS_LEG" ] && LEGS="$LEGS --gs-leg $GS_LEG"
[ -n "$G0_LEG" ] && LEGS="$LEGS --g0-leg $G0_LEG"
if { [ -n "$GS_LEG" ] && [ -z "$G0_LEG" ]; } || { [ -z "$GS_LEG" ] && [ -n "$G0_LEG" ]; }; then
  echo "REFUSING: set BOTH GS_LEG and G0_LEG or neither -- mixing an overridden sheared leg with the"
  echo "default g=0 reference silently crosses aperture families and invalidates the comparison."
  exit 1
fi

echo "### R_BLEND GAP job=$SLURM_JOB_ID ###"; date
echo "tag=$TAG   domain=${DOM:-rectangle REMIN=${REMIN:-0.3} MAGMAX=${MAGMAX:-26.0}}"
echo "legs=${LEGS:-<script defaults: non-ap7 g0.05 val / g0.0 train>}"
python -u scripts/eval_rblend_gap.py \
  --true-re-min ${REMIN:-0.3} --true-mag-max ${MAGMAX:-26.0} $DOM $LEGS \
  --tag "$TAG" \
  --output "$D/rblend_gap_${TAG}${SUF}.npz"
echo "RBLENDGAP_DONE"; date
