#!/bin/bash
#SBATCH --job-name=bfstrata
#SBATCH --time=06:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-2
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfstrata_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfstrata_%A_%a.err

# FLOW #2 RETRAIN -- fixing the faint-pair conditional bias (WORKLOG 2026-08-02e/f).
#
# THE DEFECT. On held-out pairs the flow is 2.5x OVER on the neighbours the emulator discards
# (mean 0.0083 vs a measured 0.0033) and ~11% UNDER on the ones it keeps (0.0687 vs 0.0772). Opposite
# signs, so the two largely cancel in any separation-binned average -- which is exactly what both the
# response loss and the checkpoint criterion were measuring. The model was never told about the
# neighbour-BRIGHTNESS axis, and neither was the thing choosing its checkpoint.
#
# WHY THE PER-PAIR MSE CANNOT FIX IT ON ITS OWN. The label carries std ~3.9 against a signal ~0.02,
# so the squared-error surface is dominated by irreducible noise and is nearly flat with respect to a
# systematic offset inside a stratum. Averaging ~1M pairs in a stratum drops the error to ~0.004 and
# makes that same offset a 10-sigma effect. So the fix is an ANCHOR on the stratum means, not a
# reweighting of the per-pair term.
#
# TWO CHANGES, both in scripts/train_blend_flow.py:
#   1. --strata-weight  adds chi2/dof of the per-stratum MEAN prediction against anchors measured on
#      the TRAIN split, over a separation x neighbour-magnitude grid. Each stratum is weighted by
#      1/sem^2 of its own anchor, so poorly-measured strata self-attenuate instead of needing a
#      hand-tuned weight.
#   2. --select-on strata  extends the held-out criterion to the same 2-D grid, with anchors from the
#      VAL split. Selecting on the training anchors would measure how hard the penalty pulled rather
#      than whether the model generalises.
#
# The anchors are estimated from this run's own training data and printed in this run's output --
# derived-and-reported, which is the line AGENTS.md draws. Nothing is pasted in from elsewhere and no
# constgold quantity is touched: the firewall holds, and whether this helps is judged on the RULER.
#
# THE SWEEP, sized from the smoke run (15481432) rather than guessed. The response term is normalised
# by the label variance, so it sits at res ~ 1.0 and contributes response_weight x 1.0 = 1000 to the
# loss. The strata chi2/dof came in at ~12. A weight of 1 would therefore make the penalty ~1% of the
# objective -- far too weak to move anything. 10 / 100 / 1000 brackets it from "a nudge" to "equal
# footing with the per-pair term". (An earlier version of this script swept 0.1/1/10 and would have
# wasted all three tasks on the flat end.)
#
# Note the two terms are not really comparable by magnitude: the per-pair term's value is almost all
# irreducible label noise and its USEFUL gradient is a small part of it, whereas the strata term's
# gradient is coherent. So the effective weight is higher than the ratio suggests, which is the reason
# for sweeping rather than picking.
#
# The control is the existing `blendflow_rw1k_s501` (strata weight 0) -- not retrained here, since
# nothing else changed.
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

SW_LIST=(10.0 100.0 1000.0)
SW=${SW_LIST[$SLURM_ARRAY_TASK_ID]}
TAG="rw1k_sw${SW}"
OUT=$CACHE/blendflow_${TAG}_s${SEED}.pt

echo "strata weight = $SW   -> $OUT"
python -u scripts/train_blend_flow.py \
  --pairset "$PAIRSET" --output "$OUT" \
  --epochs 40 --batch-size 65536 --lr 1e-3 \
  --mean-hidden 256 --hidden-dim 128 --n-layers 3 --n-flows 6 \
  --delta 0.02 --nll-weight 1.0 --response-weight 1000.0 \
  --strata-weight "$SW" --select-on strata \
  --val-case-frac 0.2 --patience 10 --seed "$SEED" \
  || { echo "TRAIN FAILED sw=$SW"; exit 1; }

echo "--- scoring against BlendEMU on the SAME held-out rows (the ruler, which decides) ---"
python -u scripts/eval_blend_flow.py --pairset "$PAIRSET" --checkpoint "$OUT" \
  --save-npz "$CACHE/eval_${TAG}_s${SEED}.npz" || { echo "EVAL FAILED sw=$SW"; exit 1; }

echo "BFSTRATA_DONE sw=$SW"; date
