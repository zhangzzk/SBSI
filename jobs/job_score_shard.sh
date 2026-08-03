#!/bin/bash
#SBATCH --job-name=shard
#SBATCH --time=12:00:00
#SBATCH --mem=38G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/shard_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/shard_%j.err
#
# One disjoint slice of the catalogue, scored into its own cache.  The catalogue holds
# 31.4M rows and the runs have been using 2M; sigma_gal is now the ONLY wall on the 0.3%
# deliverable (cont.177), and it falls as 1/sqrt(N).  The per-block sums are additive, so
# several shards merge exactly -- three 200-block shards give a 600-block jackknife over
# the union.  Merge with:
#   --load-scores a.npz,b.npz,c.npz
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
case "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" in
  *[0-9]Q|*[0-9]A|*[0-9]B) unset PYTORCH_CUDA_ALLOC_CONF ;;
esac
SC=/project/ls-gruen/users/zekang.zhang/sbsi_scores
TAG=$(echo "${CUT:-0.6}" | tr -d '.' | sed 's/^0//')
date; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
python -u scripts/eval_score_select.py \
  --closure-g 0.05 --cut-abs-ehat "${CUT:-0.6}" \
  --max-rows "${ROWS:-4000000}" --row-shard "${SHARD:-0}" --row-shards "${NSHARDS:-1}" \
  --pi-rows 1048576 --pi-samples 8 --pi-reps 6 \
  --ring rot90 --shape-reps 2 --jk-blocks 200 --uncut-control \
  --pi-grid-n "${PGRID:-141}" \
  --save-scores "$SC/c0${TAG}_g05_r${ROWS:-4000000}_s${SHARD:-0}of${NSHARDS:-1}_G2765.npz" 2>&1 \
  | grep --line-buffered -vE "module command|Pi rep "
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
