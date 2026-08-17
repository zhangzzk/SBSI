#!/bin/bash
#SBATCH --job-name=bfkchk
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfkchk_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfkchk_%j.err

# PRECONDITION CHECK for a crowding input on flow #2's BLEND channel (WORKLOG 2026-08-02k).
#
# The proposal is to give flow #2 the crowding block (nbr_flux_near/far/max, log_k) so it knows a
# third galaxy shares the aperture. The physical argument is sound -- a third galaxy competes for the
# same light and changes what a pair does to the primary's measured shape -- but "sound" is not
# "measured", and NEITHER model can currently see crowding: flow #2 conditions on one neighbour at a
# time, and BlendEMU's seven features are all pair-level.
#
# So the first question is not how to add the feature. It is whether there is anything to fix. This
# scores the EXISTING round-2 checkpoint, unchanged, and bins the per-pair blend response by the
# primary's neighbour count. If flow and label agree across k, a crowding input has nothing to add on
# this channel and the sweep should not be run at all.
#
# Cheap by design: no training, one scoring pass over the held-out cases (the flow response takes
# ~10s; the emulator pass dominates). Running this before the 6-task paired sweep is the difference
# between testing a mechanism and assuming one.
#
# Note the k here is counted over the pairs that SURVIVE this script's own filters, so it is the
# crowding the model was actually shown -- the same convention the pair set's stored `k` uses.
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
CKPT=$CACHE/blendflow_rw1k_grids_sw100.0_s501.pt

echo "### k-dependence check on $CKPT ###"; date
python -u scripts/eval_blend_flow.py --pairset "$PAIRSET" --checkpoint "$CKPT" \
  --save-npz "$CACHE/eval_kcheck_round2_s501.npz" || { echo "FAILED"; exit 1; }
echo "BFKCHK_DONE"; date
