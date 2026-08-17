#!/bin/bash
#SBATCH --job-name=cache_smoke
#SBATCH --time=00:40:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/cache_smoke_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/cache_smoke_%j.err
#
# Regression test for --save-scores / --load-scores: run the SAME small configuration
# twice, once scoring and once off the cache, and require the two reports to be
# character-identical apart from the two lines that are supposed to differ.  The cache
# exists so the population block can be re-estimated without a multi-hour score pass;
# that is only sound if it changes no number at all.
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

OUT=/project/ls-gruen/users/zekang.zhang/sbsi_scores
mkdir -p "$OUT"
NPZ="$OUT/smoke_$SLURM_JOB_ID.npz"
COMMON="--closure-g 0.05 --cut-abs-ehat 0.6 --max-rows 20000 --grid-n 21
        --pi-rows 512 --pi-samples 8 --pi-reps 2 --shape-reps 1 --jk-blocks 20
        --uncut-control"

date
python -u scripts/eval_score_select.py $COMMON --save-scores "$NPZ" 2>&1 \
  | grep -vE "module command" > /home/z/Zekang.Zhang/logs/cache_smoke_A.txt
RCA=$?
python -u scripts/eval_score_select.py $COMMON --load-scores "$NPZ" 2>&1 \
  | grep -vE "module command" > /home/z/Zekang.Zhang/logs/cache_smoke_B.txt
RCB=$?

# BOTH RUNS MUST SUCCEED, checked before the diff.  A diff of two outputs is a test of
# agreement, not of correctness: when a later refactor dropped `keep_frac`, both runs
# crashed at the same line and produced identical truncated reports, and this script duly
# reported CACHE OK.  Two identical failures are not a pass.
if [ "$RCA" -ne 0 ] || [ "$RCB" -ne 0 ]; then
  echo "SMOKE FAILED: run exited non-zero (save=$RCA load=$RCB); tails follow"
  tail -20 /home/z/Zekang.Zhang/logs/cache_smoke_A.txt
  tail -20 /home/z/Zekang.Zhang/logs/cache_smoke_B.txt
  date; echo "### DONE (exit 1) ###"; exit 1
fi

# Strip only what MUST differ: the progress lines, the two cache banners, and the
# timestamped filename.  Everything else is a number and must match exactly.
strip() { grep -vE "leg [0-9]+/[0-9]+\]|Pi rep |score sums cached|cached scores loaded" "$1"; }
if diff <(strip /home/z/Zekang.Zhang/logs/cache_smoke_A.txt) \
        <(strip /home/z/Zekang.Zhang/logs/cache_smoke_B.txt); then
  echo "CACHE OK: scored and cached runs are identical"
  STATUS=0
else
  echo "CACHE MISMATCH (diff above)"
  STATUS=1
fi

# The guard must also FIRE on a mismatched key.  Same cache, different cut: must exit
# non-zero rather than quietly pair these galaxies with a different selection.
if python -u scripts/eval_score_select.py $COMMON --cut-abs-ehat 0.4 \
      --load-scores "$NPZ" > /dev/null 2>&1; then
  echo "GUARD FAILED: a mismatched cache was accepted"; STATUS=1
else
  echo "GUARD OK: mismatched cache refused"
fi
date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
