#!/bin/bash
#SBATCH --job-name=bfcg100
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfcg100_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfcg100_%j.err

# constgold `m` for the STRATA-ANCHORED flow #2 (weight 100), against BlendEMU on ONE pair list.
#
# Model choice was made on the RULER (job 15481440 + control 15481586): weight 100 cuts the summed
# held-out bias from +11.03% to +1.95% and the separation chi2/dof from 1.0 to 0.2, at the cost of a
# modest degradation on the primary-property axes (3.6 -> 4.0, 2.0 -> 2.7). constgold is used here
# only to REPORT the consequence, never to select -- the R_blend firewall.
#
# Comparable prior numbers, same script, same rows (WORKLOG 2026-08-02f): control flow -2.308%,
# BlendEMU +0.279%, both on the all-pairs 7" list.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow
CKPT=$CACHE/blendflow_rw1k_sw100.0_s501.pt
CASES=$(seq 40 139)

echo "############ all pairs in 7\", both models on one list, strata-anchored checkpoint"
python -u scripts/build_blend_lookup_both.py \
    --cases $CASES --checkpoint "$CKPT" --no-restrict-to-emu-domain \
    --output results/blend_lookup_both_sw100_allpairs_c40-139.feather || exit 1

echo
echo "############ constgold m"
python -u scripts/eval_m_with_blendflow.py \
    --flow-lookup results/blend_lookup_both_sw100_allpairs_c40-139.feather \
    --emu-lookup  results/blend_lookup_both_sw100_allpairs_c40-139.feather \
    --flow-col R_blend_flow --emu-col R_blend_emu --in-domain-only || exit 1

echo "BFCG100_DONE"
