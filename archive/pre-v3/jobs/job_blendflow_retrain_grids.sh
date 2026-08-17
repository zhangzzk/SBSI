#!/bin/bash
#SBATCH --job-name=bfgrids
#SBATCH --time=06:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-2
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfgrids_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfgrids_%A_%a.err

# FLOW #2 RETRAIN, ROUND 2 -- anchoring the PRIMARY axes as well (WORKLOG 2026-08-02g "Next" (a)).
#
# WHERE ROUND 1 LEFT IT. The sep x neighbour-mag anchor cut the summed held-out bias +11.03% -> +1.95%
# and the separation chi2/dof 1.0 -> 0.2 (emulator 0.7), and constgold `m` -2.308% -> -0.495%. But
# every penalised variant came out WORSE than the control when binned by the PRIMARY's own size and
# magnitude (Re_p 3.6 -> 4.0, r_p 2.0 -> 2.7 at weight 100), against the emulator's 1.3 and 0.6. That
# is a second conditional bias, on an axis round 1's grid never named, and it is now the binding
# constraint on both the ruler and `m`.
#
# WHY MARGINAL GRIDS AND NOT ONE JOINT GRID. Crossing separation x neighbour-mag x primary-mag x
# primary-size is ~1700 cells. The realised anchor S/N is already median ~1.7 on 48 cells, so
# splitting the same pairs that far would leave nearly every anchor consistent with zero: the penalty
# would still report a large chi2 while pulling toward noise. Three 2-D grids, each marginalised over
# the axes it does not name, keep ~1M pairs per cell and constrain all four axes. Their chi2/dof are
# AVERAGED, not pooled, so an axis gains no extra weight merely by having more bins.
#
# THE SWEEP brackets round 1's winner (100). Round 1 showed the penalty is not monotone -- weight 1000
# over-corrected past the target and landed worse than the control on both primary axes -- so this
# stays inside the range that behaved, rather than pushing further.
#
# Read the result on the RULER (`eval_blend_flow.py`, run below on identical held-out rows), never on
# constgold. Anchors are estimated from this run's own TRAIN split and printed in its output.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
if [ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow
PAIRSET=$CACHE/blend_pairset_ap7.feather
SEED=501

SW_LIST=(30.0 100.0 300.0)
SW=${SW_LIST[$SLURM_ARRAY_TASK_ID]}
TAG="rw1k_grids_sw${SW}"
OUT=$CACHE/blendflow_${TAG}_s${SEED}.pt

echo "strata weight = $SW   -> $OUT"
python -u scripts/train_blend_flow.py \
  --pairset "$PAIRSET" --output "$OUT" \
  --epochs 40 --batch-size 65536 --lr 1e-3 \
  --mean-hidden 256 --hidden-dim 128 --n-layers 3 --n-flows 6 \
  --delta 0.02 --nll-weight 1.0 --response-weight 1000.0 \
  --strata-weight "$SW" --select-on strata \
  --grids sep_x_nbrmag,sep_x_primag,sep_x_prisize \
  --val-case-frac 0.2 --patience 10 --seed "$SEED" \
  || { echo "TRAIN FAILED sw=$SW"; exit 1; }

echo "--- scoring against BlendEMU on the SAME held-out rows (the ruler, which decides) ---"
python -u scripts/eval_blend_flow.py --pairset "$PAIRSET" --checkpoint "$OUT" \
  --save-npz "$CACHE/eval_${TAG}_s${SEED}.npz" || { echo "EVAL FAILED sw=$SW"; exit 1; }

echo "BFSTRATA_DONE sw=$SW"; date
