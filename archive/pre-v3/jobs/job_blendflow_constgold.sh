#!/bin/bash
#SBATCH --job-name=bfcgold
#SBATCH --time=08:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfcgold_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfcgold_%j.err

# FLOW #2 on constgold -- end to end, with the neighbour-selection confound removed.
#
# WHY THREE STEPS RATHER THAN ONE `m`. The first constgold attempt gave <R_blend> = 0.468 against the
# fiducial emulator's 0.136 and was read as "constgold is twice as dense". That was wrong. The
# emulator's stored regression config counts a neighbour only if the NEIGHBOUR itself has
# 18 <= mag < 26 and 0.3 <= Re <= 1.5 (r_max 10", k 20); everything fainter or smaller is assigned
# exactly zero. Flow #2 trained on neighbours down to mag 29 / Re 0.01, so an unrestricted sum is
# faithful to its own training but is NOT summed over the emulator's pair set. Comparing the two
# without fixing that confounds a model difference with a bookkeeping difference.
#
#   step 1  decide on the RULER whether the dropped neighbours actually blend. constgold `m` may not
#           settle this -- that is the R_blend firewall -- but the half-shear labels can.
#   step 2  build BOTH lookups from the same checkpoint: the flow's native sum, and one restricted to
#           exactly the emulator's neighbour selection.
#   step 3  score `m` for both against the fiducial emulator on identical rows.
#
# SEEDS. `m` is a bias on a shape response, so all 16 flow-#1 seeds are used and the ratio is formed
# inside each seed. This does NOT include flow #2's own seed variance -- one checkpoint is a single
# draw -- so every `m` below is a FIRST LOOK, not a quotable number. Quoting requires flow #2 to be
# ensembled over 16 seeds in its own right.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
if [ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow
CKPT=$CACHE/blendflow_rw1k_s501.pt          # best held-out chi2/dof (0.99) of the response sweep
PAIRSET=$CACHE/blend_pairset_ap7.feather
CASES=$(seq 40 139)

echo "############ STEP 1: do the emulator's dropped neighbours blend? (ruler, not constgold)"
python -u scripts/eval_faint_neighbour_contribution.py \
    --pairset "$PAIRSET" --checkpoint "$CKPT" || exit 1

echo
echo "############ STEP 2a: flow lookup, NATIVE sum (every neighbour inside 7\")"
# --allow-extrapolation is required and is NOT a way of ignoring the domain: the constgold field
# contains many primaries fainter than mag 26, the flow's `in_domain` flag marks them, and step 3
# cuts on that flag. The rows are carried flagged, never zero-filled (AGENTS.md).
python -u scripts/build_blend_lookup_flow.py \
    --cases $CASES --checkpoint "$CKPT" \
    --output results/blend_lookup_flow_native_c40-139.feather \
    --aperture 7.0 --allow-extrapolation || exit 1

echo
echo "############ STEP 2b: flow lookup, summed over the EMULATOR'S neighbour selection"
python -u scripts/build_blend_lookup_flow.py \
    --cases $CASES --checkpoint "$CKPT" \
    --output results/blend_lookup_flow_emumatch_c40-139.feather \
    --aperture 7.0 --allow-extrapolation \
    --neighbour-mag-range 18 26 --neighbour-re-range 0.3 1.5 --max-neighbours 20 || exit 1

echo
echo "############ STEP 3a: constgold m -- flow #2 NATIVE sum vs the fiducial emulator"
python -u scripts/eval_m_with_blendflow.py \
    --flow-lookup results/blend_lookup_flow_native_c40-139.feather --in-domain-only || exit 1

echo
echo "############ STEP 3b: constgold m -- flow #2 EMULATOR-MATCHED sum vs the fiducial emulator"
python -u scripts/eval_m_with_blendflow.py \
    --flow-lookup results/blend_lookup_flow_emumatch_c40-139.feather --in-domain-only || exit 1

echo "BFCGOLD_ALL_DONE"
