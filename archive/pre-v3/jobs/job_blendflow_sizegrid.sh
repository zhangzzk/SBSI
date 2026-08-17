#!/bin/bash
#SBATCH --job-name=bfsize
#SBATCH --time=08:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-5
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfsize_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfsize_%A_%a.err

# THE LAST AXIS BlendEMU STILL WINS: PRIMARY SIZE (WORKLOG 2026-08-02l).
#
# Round 2 left flow #2 ahead of the emulator on separation (0.1 vs 0.7), primary magnitude (0.4 vs
# 0.6) and summed bias, and BEHIND on primary size: chi2/dof 2.2 against 1.3, replicated at 2.0. That
# gap (~2.1 vs 1.3) is well outside the 0.2 reproducibility floor measured in 2026-08-02j, so it is a
# real defect and not a lucky draw.
#
# THE HYPOTHESIS. All three current grids cross their second axis with SEPARATION. A bias living in
# the primary-size x neighbour-BRIGHTNESS plane -- a large primary beside a faint neighbour behaving
# unlike a large primary beside a bright one AT THE SAME SEPARATION -- is marginalised away in every
# one of them. `prisize_x_nbrmag` names that plane. This is the same reasoning that made the
# neighbour-magnitude axis worth adding in round 1, where the defect was likewise invisible to a
# criterion binned only by separation.
#
# PAIRED DESIGN, because a single run cannot resolve this. Two runs of the identical configuration on
# identical data with the SAME seed differ by 0.28 pt of summed bias and ~0.2 in chi2/dof -- CUDA
# nondeterminism, since the stratum penalty uses `index_add_`. Each seed is therefore run BOTH ways
# and the statistic is the PER-SEED difference, which cancels the seed entirely. Three pairs is thin,
# but it is a measured error bar rather than an assumed one.
#
# WHAT CHANGES AND WHAT DOES NOT. Only the grid list. Weights stay at the round-2 winner (response
# 1000, strata 100), selection stays on strata, features stay the 13 (no crowding block -- the
# 2026-08-02k check found nothing for it to fix on this channel). Changing a weight at the same time
# would leave no way to attribute the result.
#
# NOTE the penalty averages chi2/dof ACROSS grids, so the fourth grid dilutes each of the others by
# 3/4 as well as adding its own constraint. If the size axis improves while separation degrades, that
# dilution -- not the new axis -- is the first thing to suspect, and the fix would be to weight the
# grids rather than to add more.
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

G3="sep_x_nbrmag,sep_x_primag,sep_x_prisize"
G4="sep_x_nbrmag,sep_x_primag,sep_x_prisize,prisize_x_nbrmag"

SEEDS=(501 501 502 502 503 503)
ARMS=(g3 g4 g3 g4 g3 g4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
if [ "$ARM" = "g4" ]; then GRIDS="$G4"; else GRIDS="$G3"; fi

TAG="sizegrid_${ARM}"
OUT=$CACHE/blendflow_${TAG}_s${SEED}.pt

if [ ! -f "$PAIRSET" ]; then
  echo "MISSING $PAIRSET -- run jobs/job_blend_pairset_dual.sh first"; exit 1
fi

echo "arm=$ARM seed=$SEED grids=$GRIDS  -> $OUT"
python -u scripts/train_blend_flow.py \
  --pairset "$PAIRSET" --output "$OUT" \
  --epochs 40 --batch-size 65536 --lr 1e-3 \
  --mean-hidden 256 --hidden-dim 128 --n-layers 3 --n-flows 6 \
  --delta 0.02 --nll-weight 1.0 --response-weight 1000.0 \
  --strata-weight 100.0 --select-on strata --grids "$GRIDS" \
  --val-case-frac 0.2 --patience 10 --seed "$SEED" \
  || { echo "TRAIN FAILED $ARM s$SEED"; exit 1; }

echo "--- the ruler: flow vs BlendEMU on the SAME held-out rows ---"
python -u scripts/eval_blend_flow.py --pairset "$PAIRSET" --checkpoint "$OUT" \
  --save-npz "$CACHE/eval_${TAG}_s${SEED}.npz" || { echo "EVAL FAILED $ARM s$SEED"; exit 1; }

echo "BFSIZE_DONE arm=$ARM seed=$SEED"; date
