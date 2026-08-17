#!/bin/bash
#SBATCH --job-name=bfmatch
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfmatch_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfmatch_%j.err

# FLOW #2 vs BlendEMU on ONE pair list -- removing the confound WORKLOG 2026-08-02e identified.
#
# Job 15481033 could not attribute its own result: the flow read +14.2% against the emulator on the
# native sum and -22.3% on the "matched" sum, because the matched version still used the flow's 7"
# aperture against the emulator's native 10", and because the matched selection changed the row set
# (22% of primaries lost every neighbour, and the emulator's own m moved -0.126% -> +0.830%).
#
# Here both models are scored PAIR FOR PAIR on the same 7" list -- the emulator through
# `predict_on_pairs`, which scores the rows it is handed rather than re-deriving its own neighbour
# list. The aperture is 7" for both because the flow has never seen a wider pair; bringing the
# emulator down is free, extrapolating the flow up is not. The 7-10" annulus is simply outside the
# comparison, for both, rather than credited to either.
#
# Two pair sets, because they answer different questions:
#   RESTRICTED   pairs inside the emulator's own cuts -- neither model extrapolates. The clean
#                model-vs-model comparison.
#   ALL          every pair inside 7". The emulator IS extrapolated here (81.4% of these pairs are
#                outside its cuts, where it natively returns zero), so read its column as "what the
#                emulator would say if asked", not as its calibrated prediction.
#
# SEEDS: 16 flow-#1 seeds, ratio formed inside each seed. Flow #2 is still ONE checkpoint, so the
# quoted error remains a lower bound and nothing here is quotable as certified.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
if [ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ]; then
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow
CKPT=$CACHE/blendflow_rw1k_s501.pt
CASES=$(seq 40 139)

echo "############ SMOKE: 2 cases, both models, exercise every code path before the 100-case run"
python -u scripts/build_blend_lookup_both.py \
    --cases 40 41 --checkpoint "$CKPT" \
    --output "$CACHE/blend_lookup_both_SMOKE.feather" || exit 1

echo
echo "############ RESTRICTED: pairs inside the emulator's own cuts -- neither model extrapolates"
python -u scripts/build_blend_lookup_both.py \
    --cases $CASES --checkpoint "$CKPT" \
    --output results/blend_lookup_both_emudomain_c40-139.feather || exit 1

echo
echo "############ m -- RESTRICTED pair set, both models, identical rows"
python -u scripts/eval_m_with_blendflow.py \
    --flow-lookup results/blend_lookup_both_emudomain_c40-139.feather \
    --emu-lookup  results/blend_lookup_both_emudomain_c40-139.feather \
    --flow-col R_blend_flow --emu-col R_blend_emu --in-domain-only || exit 1

echo
echo "############ ALL PAIRS in 7\" -- the emulator is extrapolated outside its cuts, read as such"
python -u scripts/build_blend_lookup_both.py \
    --cases $CASES --checkpoint "$CKPT" --no-restrict-to-emu-domain \
    --output results/blend_lookup_both_allpairs_c40-139.feather || exit 1

echo
echo "############ m -- ALL-PAIR set, both models, identical rows"
python -u scripts/eval_m_with_blendflow.py \
    --flow-lookup results/blend_lookup_both_allpairs_c40-139.feather \
    --emu-lookup  results/blend_lookup_both_allpairs_c40-139.feather \
    --flow-col R_blend_flow --emu-col R_blend_emu --in-domain-only || exit 1

echo "BFMATCH_ALL_DONE"
