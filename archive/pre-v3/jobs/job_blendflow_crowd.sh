#!/bin/bash
#SBATCH --job-name=bfcrowd
#SBATCH --time=08:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-5
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfcrowd_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfcrowd_%A_%a.err

# DOES A CROWDING INPUT HELP FLOW #2's BLEND CHANNEL? (WORKLOG 2026-08-02k)
#
# ONLY RUN THIS IF `jobs/job_blendflow_kcheck.sh` SHOWED A k-DEPENDENT BIAS. That check scores the
# existing round-2 checkpoint and bins the per-pair blend response by neighbour count. If flow and
# label already agree across k, there is nothing here for a crowding feature to fix and this sweep
# would be measuring noise.
#
# PAIRED DESIGN, and the pairing is the point. An accidental replicate (WORKLOG 2026-08-02j) showed
# that two runs of the IDENTICAL configuration on IDENTICAL data with the SAME seed differ by 0.28
# points of summed bias and ~0.2 in chi2/dof -- CUDA nondeterminism, since the stratum penalty uses
# `index_add_`. A single control against a single crowding run therefore cannot resolve an effect
# smaller than that. Here each seed is run BOTH ways and the comparison is the per-seed difference,
# which cancels the seed and leaves only the feature block. Three pairs is thin but it is a real
# error bar instead of an assumed one.
#
# BLEND-ONLY: no --self-response-weight. The merge is parked (owner's call, 2026-08-02j); this uses
# only the piece of that work that serves the separate blend model.
#
# WHAT CHANGES AND WHAT DOES NOT. Only `--crowding`. The grids stay sep x nbrmag / primag / prisize,
# the weights stay at the round-2 winner (response 1000, strata 100), selection stays on strata.
# Adding a `sep x k` anchor at the same time would confound the feature with the criterion, and there
# would be no way to say which one moved the result.
#
# DEPLOYMENT CAVEAT, recorded before the fact: if the crowding arm wins,
# `build_blend_lookup_flow.py` and `build_blend_lookup_both.py` currently REFUSE such a checkpoint --
# their per-case frames carry no `pid`/`k`, and rebuilding the block from whatever rows are present
# would describe a systematically under-crowded galaxy. Threading `pid`/`k` through those two is then
# required before any constgold `m` can be produced from this model.
#
# FIREWALL: half-shear only. constgold is not opened here.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
if [ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow
PAIRSET=$CACHE/blend_pairset_ap7_dual.feather

SEEDS=(501 501 502 502 503 503)
CROWD=(0 1 0 1 0 1)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
C=${CROWD[$SLURM_ARRAY_TASK_ID]}

CROWD_FLAG=""; ARM="plain"
if [ "$C" = "1" ]; then CROWD_FLAG="--crowding"; ARM="crowd"; fi
TAG="crowdtest_${ARM}"
OUT=$CACHE/blendflow_${TAG}_s${SEED}.pt

if [ ! -f "$PAIRSET" ]; then
  echo "MISSING $PAIRSET -- run jobs/job_blend_pairset_dual.sh first"; exit 1
fi

echo "arm=$ARM seed=$SEED  -> $OUT"
python -u scripts/train_blend_flow.py \
  --pairset "$PAIRSET" --output "$OUT" \
  --epochs 40 --batch-size 65536 --lr 1e-3 \
  --mean-hidden 256 --hidden-dim 128 --n-layers 3 --n-flows 6 \
  --delta 0.02 --nll-weight 1.0 --response-weight 1000.0 \
  --strata-weight 100.0 --select-on strata $CROWD_FLAG \
  --val-case-frac 0.2 --patience 10 --seed "$SEED" \
  || { echo "TRAIN FAILED $ARM s$SEED"; exit 1; }

echo "--- the ruler: flow vs BlendEMU on the SAME held-out rows ---"
python -u scripts/eval_blend_flow.py --pairset "$PAIRSET" --checkpoint "$OUT" \
  --save-npz "$CACHE/eval_${TAG}_s${SEED}.npz" || { echo "EVAL FAILED $ARM s$SEED"; exit 1; }

echo "BFCROWD_DONE arm=$ARM seed=$SEED"; date
