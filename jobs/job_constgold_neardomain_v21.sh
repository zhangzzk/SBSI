#!/bin/bash
#SBATCH --job-name=cgnd_v21
#SBATCH --time=10:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/cgnd_v21_%j.out

# constgold SELECTION table for the V2.1 model, on REAL per-leg measured mag/size. Sim and model cut
# on the SAME quantity at the SAME absolute threshold -- no proxy except the (*) real-S/N rows,
# which the flow cannot represent because it does not output FLUX_AUTO/FLUXERR_AUTO.
#
# ONE LEVER vs jobs/job_constgold_neardomain.sh (the fiducial table): the model. Checkpoints are the
# V2.1 flows, the domain is the V2.1 curve, and R_blend comes from the V2.1 emulator. Same script,
# same catalogue, same --min-case 40, same --max-rows 0.
#
# WHY --v21-domain AND NOT --dom-mag-max/--dom-re-min. The V2.1 domain is true Re > 0.5" AND true
# S/N > 10; the S/N half is a CURVE in (mag, Re), so no rectangle expresses it. Scoring a flow
# outside its training domain is not a mild extrapolation -- it put the V2.1 flow's
# certified-convention m at -42.6% (job 15523254). sbs_shear.domain holds the single definition.
#
# WHY 16 SEEDS. This table reports `m`, a bias on the SHAPE (e) response, and AGENTS.md sets the
# e-response standard at 16 regardless of what the CUTS are on. The cuts here are measured
# flux/size, which is the 4-seed standard, but they only decide WHICH objects enter the average --
# the number reported is the shape response of that subset. V2.1 is also noisier per seed than the
# fiducial (sd 1.014% vs 0.606%), so 16 is the floor here, not a luxury. Do not quote m from fewer.
#
# READ `dm`, NOT ONLY `m`. The per-seed offset cancels in model-vs-model ratios (column 4) and in
# dm = m(cut) - m(no cut), but NOT in the absolute m at a cut, which is model-vs-SIM and inherits
# the full no-cut seed error. On V2.1 that no-cut error is ~+-0.25% at 16 seeds.
#
# CUT CHOICE. Measured flux_radius is in ARCSEC and is floored by the PSF (R50 = 0.527"), so size
# cuts below ~0.5" keep ~100% and are no-ops BY CONSTRUCTION -- the list therefore starts at 0.55"
# and runs out to 1.00", which is aggressive against a domain whose floor is a TRUE Re of 0.5".
# Magnitude runs 26.0 (essentially no-op inside the domain, whose S/N>10 edge is mag 25.72 at
# Re=0.5") down to 24.0 (aggressive). The point of going past the domain edge is that these are
# MEASURED cuts: unlike true-property cuts they move with shear, so the moving-boundary term is
# live and column (1) is non-trivial.
#
# THE [T] TRUE-CUT ROWS ARE THE CONTROL. True mag/Re are conditioning INPUTS, not flow outputs, so
# both sides select IDENTICAL objects and dm is a pure response error with no selection-modelling
# channel. If the measured rows fail where the matched true rows pass, the failure is selection
# modelling; if both fail, it is the response.
#
# FIREWALL: constgold is EVALUATION ONLY. Nothing here trains, tunes or selects a model.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
if [ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
# The V2.1 blend lookup lives in the WORKTREE's gitignored results/, not the main checkout's
# (sbs_shear/paths.py records the split). Pointing at $HOME/SBSI/results cost a failed run (15522857).
LOOKUP=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/results/blend_lookup_v21_c40-139.feather

SEEDS_EXPLICIT=${SEEDS:+yes}   # capture BEFORE defaulting: an explicit SEEDS is the opt-out signal
SEEDS="${SEEDS:-501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517}"
CK=""; MISSING=""
for sd in $SEEDS; do
  f=$D/measurement_flow_g0_ngmix_ablate_s2c_lt500_v21_s${sd}_swaavg.pt
  if [ -f "$f" ]; then CK="$CK $f"; else MISSING="$MISSING s$sd"; fi
done
NCK=$(echo $CK | wc -w)
echo "### CONSTGOLD NEAR-DOMAIN V2.1 job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
echo "checkpoints found: $NCK   missing:${MISSING:- none}"
echo "blend lookup: $LOOKUP"
[ -f "$LOOKUP" ] || { echo "MISSING PREREQUISITE: $LOOKUP"; exit 1; }
# Never QUIETLY produce an under-seeded m. 16 is the e-response standard; fewer is allowed only as a
# deliberate, explicit choice (set SEEDS), and then the run is banner-marked as dm-only.
if [ "$NCK" -lt 16 ]; then
  if [ -z "$SEEDS_EXPLICIT" ]; then
    echo "REFUSING: only $NCK checkpoints and SEEDS was not set explicitly. This table reports m,"
    echo "which needs 16 seeds (AGENTS.md). Pass SEEDS=... to opt into a smaller, dm-only run."
    exit 1
  fi
  cat <<BANNER

################################################################################
##  DM-ONLY RUN -- $NCK SEEDS. DO NOT QUOTE THE ABSOLUTE m COLUMN FROM THIS RUN.
##
##  AGENTS.md: "Do not quote any m from 4 seeds." The absolute m at a cut is
##  model-vs-SIM; the sim side has no seed dependence, so each seed's own offset
##  survives in full and m inherits the whole no-cut seed error. On V2.1 the
##  per-seed sd is 1.014%, so at $NCK seeds that is roughly +-$(awk -v n=$NCK 'BEGIN{printf "%.2f", 1.014/sqrt(n)}')%.
##
##  READ dm = m(cut) - m(no cut), AND column (4) (model-vs-model). Both terms
##  move together seed to seed, so the offset cancels and these ARE valid here.
##  The [T] true-cut rows remain a valid control for the same reason.
################################################################################

BANNER
fi

python -u scripts/eval_selection_constgold_neardomain.py \
  --ckpt $CK --v21-domain --blend-lookup "$LOOKUP" \
  --n-samples 32 --batch-size 16384 --max-rows 0 --min-case 40 \
  --mag-cuts 26.0 25.5 25.0 24.5 24.0 \
  --size-cuts 0.55 0.60 0.70 0.80 1.00 \
  --sn-cuts 10.0 15.0 20.0 \
  --combo-cuts "25.5:0.55" "25.0:0.60" "24.5:0.70" "24.0:0.80" \
  --true-cuts --true-mag-cuts 25.5 25.0 24.5 --true-re-cuts 0.6 0.8 \
  --true-combo-cuts "25.0:0.6" \
  --save-npz results/constgold_neardomain_v21_table.npz \
  2>&1 | grep -v --line-buffered "module command" || { echo CGND_V21_FAILED; exit 1; }
echo CGND_V21_ALL_DONE; date
