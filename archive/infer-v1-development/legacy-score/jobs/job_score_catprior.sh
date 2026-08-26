#!/bin/bash
# Archived Infer V1 development job.
# LEGACY: finite-M score/quadrature experiment.  New closure jobs call
# scripts/run_catalogue_closure.py through the job_catalogue_* wrappers.
#SBATCH --job-name=catprior
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/catprior_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/catprior_%j.err
#
# §5B closure with the conditioning prior DRAWN FROM THE INPUT CATALOGUE rather than
# pinned at each row's own truth.  Both likelihoods are exact for these data, so the
# difference between them is pure finite-draw quadrature error and must fall as 1/M.
#
# Submit:
#   FEAT=nbr sbatch $(jobs/pick_gpu.sh) jobs/job_score_catprior.sh
#   FEAT=all sbatch $(jobs/pick_gpu.sh) jobs/job_score_catprior.sh
#   FEAT=all BETA=0.8 sbatch $(jobs/pick_gpu.sh) jobs/job_score_catprior.sh
#
# Pool halves (cont.189).  Both arms MUST share FEAT/ROWS/LADDER/DRAWSEED and the split
# seed, and differ only in HALF -- the shared draw seed is what makes the difference paired:
#   HALF=A SPLIT=1 sbatch $(jobs/pick_gpu.sh) jobs/job_score_catprior.sh
#   HALF=B SPLIT=1 sbatch $(jobs/pick_gpu.sh) jobs/job_score_catprior.sh
# Several SPLIT values are required before anything is quotable: one split is n = 1.
#
# k-WAY SHARDS (cont.189 §6c) -- PREFER THIS over halves.  k arms give k-1 degrees of
# freedom from k jobs, where halves give 1 from 2, and the finite-catalogue term is k times
# larger per arm so the jackknife floor matters less.  8 jobs as k=8 reach 1.8 sigma where
# 8 jobs as 4 half-splits reach 1.1.  Run ALL k shards of one SPLIT:
#   for i in $(seq 0 7); do SHARD=$i NSHARDS=8 SPLIT=1 sbatch $(jobs/pick_gpu.sh) jobs/job_score_catprior.sh; done
# Two different NSHARDS values test the a ~ 1/N_pool scaling that one k has to assume.
#
# Cost is (max(LADDER) + 1) flow passes per (row, node) -- the +1 is the pinned control,
# which has to run on the SAME xhat or the comparison stops being paired.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
# vGPU slices lack the CUDA VMM APIs the expandable allocator needs (see job_score_select.sh)
case "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" in
  *[0-9]Q|*[0-9]A|*[0-9]B) unset PYTORCH_CUDA_ALLOC_CONF ;;
esac

FEAT=${FEAT:-nbr}
LADDER=${LADDER:-1,2,4,8,16}
BETA=${BETA:-0.0}
CELLS=${CELLS:-8}
ROWS=${ROWS:-120000}
G=${G:-0.05}
GRIDN=${GRIDN:-61}
DRAWSEED=${DRAWSEED:-99}
CHUNK=${CHUNK:-512}
# The ladder's top rung is part of the identity of a run, not a detail: two runs that
# differ only in it produce different nested prefixes and must not share a cache name.
MMAX=${LADDER##*,}
# POOL HALF/SHARD BELONGS IN THE FILENAME.  Arms A and B differ only in --pool-half, so
# without this they would write to the SAME path and the second would silently overwrite
# the first -- destroying the pair the run exists to produce.  Same reasoning as MMAX above.
HALF=${HALF:-none}
SPLIT=${SPLIT:-4242}
SHARD=${SHARD:--1}
NSHARDS=${NSHARDS:-2}
HALFTAG=""
if [ "$SHARD" -ge 0 ]; then
  HALFTAG="_sh${SHARD}of${NSHARDS}s${SPLIT}"
elif [ "$HALF" != "none" ]; then
  HALFTAG="_half${HALF}s${SPLIT}"
fi
OUT=${OUT:-/project/ls-gruen/users/zekang.zhang/sbsi_catprior/catprior_${FEAT}_b${BETA}_d${DRAWSEED}_r${ROWS}_G${GRIDN}_M${MMAX}${HALFTAG}.npz}
mkdir -p "$(dirname "$OUT")"

date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "FEAT=$FEAT LADDER=$LADDER BETA=$BETA CELLS=$CELLS ROWS=$ROWS G=$G DRAWSEED=$DRAWSEED HALF=$HALF SHARD=$SHARD NSHARDS=$NSHARDS SPLIT=$SPLIT"
stdbuf -oL -eL python -u scripts/eval_score_catprior.py \
  --features "$FEAT" --ladder "$LADDER" --beta "$BETA" --cells "$CELLS" \
  --max-rows "$ROWS" --closure-g "$G" --grid-n "$GRIDN" \
  --draw-seed "$DRAWSEED" --chunk "$CHUNK" \
  --pool-half "$HALF" --pool-shard "$SHARD" --pool-nshards "$NSHARDS" \
  --pool-split-seed "$SPLIT" \
  --load-oversample "${OVERSAMPLE:-7.0}" --shape-reps "${SHAPEREPS:-1}" \
  --jk-blocks "${JKBLOCKS:-200}" --save "$OUT" ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
