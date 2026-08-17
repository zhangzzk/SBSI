#!/bin/bash
#SBATCH --job-name=sh_merge
#SBATCH --partition=cluster
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --output=/home/z/Zekang.Zhang/logs/sh_merge_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/sh_merge_%j.err
#
# Cash in finished row shards without touching a GPU.  The per-block sums are already on disk,
# so `--load-scores a,b,c` needs NO score pass; the only real work left is the population (Pi)
# block, and that is small enough to survive on CPU.  The point is the QOS split: `cip` caps
# this account at 3 GPUs (and they are usually held by the flow/cg arrays), while `cluster` is
# a separate QOS with 50 job slots, so this contends with nothing.
#
# Sizing, learned the hard way: job 15527903 asked for 3 h and was killed by the time limit
# having produced no result at all.  The Pi block at pi_grid_n=141 costs ~10-15 min on an
# a40-16gb slice, and the measured CPU:GPU ratio on this pass is ~13x, which lands at 2-3.5 h
# -- right on top of a 3 h wall.  12 h on `cluster` costs nothing and removes the guesswork.
#
# 32 CPUs deliberately, not 64: the benchmark measured 65.0 rows/s on 32 cores and 57.6 on 64,
# i.e. it stops scaling and then gets WORSE (memory-bandwidth bound), so a bigger ask would
# only queue longer for less throughput.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
NT=${SLURM_CPUS_PER_TASK:-32}
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT
SC=/project/ls-gruen/users/zekang.zhang/sbsi_scores
CUT=${CUT:-0.6}
TAG=$(echo "$CUT" | tr -d '.' | sed 's/^0//')
CG=${CLOSURE_G:-0.05}
GTAG="g$(printf '%02d' "$(python -c "print(round(float('$CG')*100))")")"
# Default to the three shards that survived the queue clear-out (1, 3, 5 of 6) = 6M rows.
SHARDS=${SHARDS:-1 3 5}
NSH=${NSHARDS:-6}
ROWS=${ROWS:-2000000}
LIST=""
for S in $SHARDS; do
  F="$SC/c0${TAG}_${GTAG}_r${ROWS}_s${S}of${NSH}_G2765.npz"
  [ -f "$F" ] || { echo "MISSING shard cache: $F"; exit 1; }
  LIST="${LIST:+$LIST,}$F"
done
date; echo "host=$(hostname) threads=$NT"
echo "cut=$CUT closure_g=$CG shards={$SHARDS} of $NSH  ->  $(echo "$SHARDS" | wc -w) x $ROWS rows"
# `--line-buffered` matters: without it grep holds ~4 kB and a job killed at the wall writes a
# log with nothing in it but the headers printed before the pipe opened.
python -u scripts/eval_score_select.py \
  --closure-g "$CG" --cut-abs-ehat "$CUT" --max-rows "$ROWS" \
  --row-shards "$NSH" --pi-grid-n "${PGRID:-141}" --device cpu \
  --pi-rows 1048576 --pi-samples 8 --pi-reps 6 \
  --ring rot90 --shape-reps 2 --jk-blocks 200 --uncut-control \
  --load-scores "$LIST" 2>&1 \
  | grep --line-buffered -vE "module command|Pi rep "
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
