#!/bin/bash
#SBATCH --job-name=score_select
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/score_select_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/score_select_%j.err
#
# §5B end to end WITH selection: the full (5.3) on the real flow against a known shear.
# Closes cont.164 defect 2 (no I_sel) on real data; A.7 prices its omission at m = -64%.
# Every supported cut is isotropic in the shape angle, so §5B.2 predicts <s>_sel ~ 0 and
# I_sel carries the whole correction.
#
# Submit:
#   sbatch $(jobs/pick_gpu.sh) jobs/job_score_select.sh                 # |xhat| < 0.6
#   CUT=none BOUND=measured_mag_auto::24.5 \
#     sbatch $(jobs/pick_gpu.sh) jobs/job_score_select.sh               # measured mag cut
#   CUT=none BOUND=measured_log_flux_radius:1.45: \
#     sbatch $(jobs/pick_gpu.sh) jobs/job_score_select.sh               # measured size cut
#
# The three keep ~70% each on the V3 domain, so they differ by cut CHANNEL and not by
# sample size.  BOUND is a single --cut-bound; pass several through EXTRA if needed.
#
# `pick_gpu.sh` takes the fastest card with a free unit.  That is the single biggest lever on
# this script: an H200 does the score pass 4.8x faster than a `cip` A40-16Q slice in plain
# fp32, with no numerical change -- more than any precision setting buys, and safely.  To
# reproduce a specific published number, pin the card instead: the CUDA RNG stream depends on
# the SM count, so a different card draws a different (equally valid) random realisation.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
# vGPU slices (their names end in a profile letter, e.g. "NVIDIA A40-16Q") lack the CUDA VMM
# APIs the expandable allocator needs.  Detected from the card we actually got, rather than
# passed in as a flag, so that picking the card dynamically -- jobs/pick_gpu.sh -- cannot
# leave a stale setting behind or need the caller to remember one.
case "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" in
  *[0-9]Q|*[0-9]A|*[0-9]B) unset PYTORCH_CUDA_ALLOC_CONF ;;
esac

G=${G:-0.05}
CUT=${CUT:-0.6}                 # `none` disables the |xhat| cut (e.g. for a mag-only cut)
BOUND=${BOUND:-}                # NAME:LO:HI on a flow OUTPUT; needs the 4-output V3 flow
ROWS=${ROWS:-400000}
# The V3 flow was trained on a NARROWED primary domain (true mag < 25.8, Re > 0.5"), which
# keeps ~18% of raw catalogue rows against the default domain's ~89%.  eval_score_select.py
# reads that domain from the checkpoint and restricts the rows to match, so the reader has
# to be told to read ~7 raw rows per row wanted or the job silently scores a short sample.
OVERSAMPLE=${OVERSAMPLE:-7.0}
PIROWS=${PIROWS:-4096}          # comma list sweeps the population sample off ONE score pass
PISAMP=${PISAMP:-8}
RING=${RING:-rot90}             # 90-degree ring pairs: the shape-noise variance reduction

date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/eval_score_select.py \
  --closure-g "$G" --cut-abs-ehat "$CUT" --max-rows "$ROWS" \
  --load-oversample "$OVERSAMPLE" ${BOUND:+--cut-bound "$BOUND"} \
  --pi-rows "$PIROWS" --pi-samples "$PISAMP" --pi-reps "${PIREPS:-4}" \
  --ring "$RING" --shape-reps "${SHAPEREPS:-1}" \
  --jk-blocks "${JKBLOCKS:-200}" --uncut-control ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
