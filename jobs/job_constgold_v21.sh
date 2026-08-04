#!/bin/bash
#SBATCH --job-name=cg_v21
#SBATCH --time=03:00:00
#SBATCH --mem=34G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40-24gb:1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_v21_s%a_%j.out

# STEP 6 of V2.1: score the V2.1 flow on constgold and report m inside the V2.1 domain.
#
# ONE LEVER vs jobs/job_s2c_domain_eval.sh (which produced the fiducial dumps): the checkpoint and
# the R_blend lookup are the V2.1 ones. Same script, same catalogue, same --min-case 40, same
# sampling settings, so a difference in m is the model and not the harness.
#
# THE DUMP WRITES TO ITS OWN DIRECTORY. The fiducial dumps in derisk/v2_domain_dumps are the
# baseline every V2.1 number is compared against; overwriting one would destroy the comparison.
#
# WHY THE COVERAGE GUARD MATTERS HERE. `validate_constant_with_blend.py` joins the R_blend lookup
# and then fillna(0.0), so a galaxy the emulator never scored is indistinguishable from a genuinely
# isolated one -- both read R_blend = 0. That is AGENTS.md "Two traps" #1, worth +28.9% of spurious
# m when it bites. `eval_v2_indomain_m.py --v21-domain` therefore checks STRUCTURALLY, before
# printing any m, that every evaluated row lies inside the V2.1 emulator's stored inference box,
# and exits if not. It should pass by construction (the box IS the V2.1 bounding box); a failure
# means the emulator config and sbs_shear.domain have drifted apart.
#
# FIREWALL: constgold is evaluation-only. Nothing here trains, tunes or selects a model.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
if [ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

SEEDS=(${SEEDS:-501 502 503 505 506 507 508 509})
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}; else SEED=${SEED:-501}; fi
TAG=${TAG:-ablate_s2c_lt500_v21}
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
RES=/home/z/Zekang.Zhang/SBSI/results
DUMPDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v21_domain_dumps
mkdir -p $DUMPDIR
CK=$D/measurement_flow_g0_ngmix_${TAG}_s${SEED}_swaavg.pt
LOOKUP=$RES/blend_lookup_v21_c40-139.feather
for f in "$CK" "$LOOKUP"; do
  [ -f "$f" ] || { echo "MISSING PREREQUISITE: $f"; exit 1; }
done

echo "### V2.1 CONSTGOLD EVAL seed=$SEED job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
echo "checkpoint: $CK"; echo "lookup    : $LOOKUP"
python -u scripts/validate_constant_with_blend.py \
  --measurement-model "$CK" --catalogue "$CAT" --min-case 40 \
  --blend-lookup "$LOOKUP" \
  --crowd-flux-lookup "$RES/crowd_flux_conc_c0-199.feather" \
  --global-only --flow-seed 12345 --n-samples 64 --max-rows 45000000 --batch-size 16384 \
  --dump "$DUMPDIR/${TAG}_perobj_s${SEED}.feather" || { echo "FAILED eval seed=$SEED"; exit 1; }

echo; echo "### m under the V2.1 domain (and the V2 conventions, for the comparison) ###"
python -u scripts/eval_v2_indomain_m.py \
  --dump-glob "$DUMPDIR/${TAG}_perobj_s${SEED}.feather" \
  --catalogue "$CAT" --min-case 40 --re-min 0.3 --mag-max 26.0 \
  --v21-domain --emulator-tag lsst_r_extnbr_v21 || { echo "FAILED m seed=$SEED"; exit 1; }

echo; echo "### THE SAME MASK ON THE FIDUCIAL DUMPS: isolates POPULATION from MODEL ###"
# V2.1 is a strict subset of V2 (100.00%), so the fiducial dom6x6 dumps can be re-masked to the
# V2.1 domain. Comparing THAT to the line above separates "the domain changed" from "the model
# changed" -- the decomposition AGENTS.md insists on, and without which a ~5% m says nothing about
# its own cause. NOTE the fiducial dumps carry the fiducial emulator's R_blend, so this row is the
# V2 MODEL on the V2.1 POPULATION, not a V2.1 model.
python -u scripts/eval_v2_indomain_m.py \
  --dump-glob "/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps/ablate_s2c_lt500_dom6x6_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40 --re-min 0.3 --mag-max 26.0 \
  --v21-domain --emulator-tag lsst_r_extnbr_indom_tuned \
  || echo "(fiducial re-mask failed -- expected if the fiducial emulator's box does not cover V2.1)"
echo "CG_V21_DONE seed=$SEED"; date
