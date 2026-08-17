#!/bin/bash
#SBATCH --job-name=cgindist
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
# time/mem sized from measured usage, NOT padded: sacct on the completed runs gives 25 min elapsed
# and MaxRSS 87 GB. The old 3h/180G request was ~7x/2x over, and on a GPU-saturated partition the
# oversized WALL CLOCK is what hurts -- backfill can only slot a job into a gap at least as long as
# its declared limit, so asking for 3h to do 25 min of work made us un-backfillable.
#SBATCH --gres=gpu:a40:1
#SBATCH --array=0-7%4
# LOOKUP=<abs path> overrides the lookup entirely (e.g. the certified one, as a same-seed control)
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cgindist_s%a_%j.out

# Confirm the mean-based m-shift estimate against the REAL pipeline. Identical to
# jobs/job_v2_constgold_8seed.sh in every respect except --blend-lookup, which points at the
# input-frame-distance emulator's lookup instead of the certified one. Two seeds (not eight): the
# question is whether the -0.90 point shift predicted by scripts/estimate_m_shift.py is real, and
# the certified per-seed scatter is far below that.
#
# WORKLOG 2026-07-28n: the corrected emulator is measurably MORE accurate per pair (in-domain
# R_blend error -11.93% -> -8.24%, close pairs -41.5% -> -31.1%) yet <R_blend> rises only +1.34%
# over the full neighbour sum, which the identity says moves m from +0.245% to about -0.655%.
# If that holds, the certified m depended on a compensating error and adopting the fix requires
# revisiting R_flow -- a result about the PIPELINE, not about the emulator.
# constgold is EVAL-ONLY (firewall): the flow trained on det_meas half-shear, the emulator on the
# response catalogue, neither on constgold.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
RES=/home/z/Zekang.Zhang/SBSI/results
LOOK=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/results
DUMPDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/indist_constgold_dumps
mkdir -p $DUMPDIR

SEEDS=(${SEEDS:-501 502 503 505 506 507 508 509})   # override from the submit line for the 510-517 half
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
# TAG selects the flow. Default = the full-population V2. Override for the CUT-POPULATION V2, whose
# checkpoints are ablate_s2c_lt500_dom6x6 (trained with --primary-mag-max 26 --primary-re-min 0.3).
TAG=${TAG:-ablate_s2c_coupling_lt500}
CK=$D/measurement_flow_g0_ngmix_${TAG}_s${S}_swaavg.pt
if [ ! -f "$CK" ]; then echo "MISSING checkpoint $CK"; exit 1; fi

echo "### CONSTGOLD seed=$S tag=$TAG job=$SLURM_JOB_ID  lookup=${SUFFIX:-indist} ###"; date
python -u scripts/validate_constant_with_blend.py \
  --measurement-model "$CK" --catalogue "$CAT" --min-case 40 \
  --blend-lookup "${LOOKUP:-$LOOK/blend_lookup_${SUFFIX:-indist}_c40-139.feather}" \
  --crowd-flux-lookup "$RES/crowd_flux_conc_c0-199.feather" \
  --global-only --flow-seed 12345 --n-samples 64 --max-rows 45000000 --batch-size 16384 ${EXTRA_ARGS} \
  --dump "$DUMPDIR/${DUMPTAG:-${SUFFIX:-indist}}_perobj_s${S}.feather" 2>&1 | grep -v --line-buffered "module command"
echo "CGINDIST_DONE seed=$S"; date
