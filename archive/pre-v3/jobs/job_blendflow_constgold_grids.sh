#!/bin/bash
#SBATCH --job-name=bfcg2
#SBATCH --time=04:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/bfcg2_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/bfcg2_%j.err

# constgold `m` for the ROUND-2 flow #2 -- all three marginal grids anchored, weight 100.
#
# Chosen on the RULER (array 15482220 + control 15481586). Round 2 beats BlendEMU on three of four
# held-out measures: summed bias -1.14% vs -1.58%, separation chi2/dof 0.1 vs 0.7, primary magnitude
# 0.4 vs 0.6; it loses only on primary size, 2.2 vs 1.3 (control was 3.6). constgold reports the
# consequence and never selects -- the R_blend firewall.
#
# Comparable prior numbers, same script, same rows, same all-pairs 7" list:
#   control flow      m = -2.308%   (WORKLOG 2026-08-02f)
#   round-1 flow      m = -0.495%   (WORKLOG 2026-08-02g)
#   BlendEMU          m = +0.279%
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

CACHE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow
CKPT=$CACHE/blendflow_rw1k_grids_sw100.0_s501.pt
CASES=$(seq 40 139)

echo "############ all pairs in 7\", both models on one list, strata-anchored checkpoint"
python -u scripts/build_blend_lookup_both.py \
    --cases $CASES --checkpoint "$CKPT" --no-restrict-to-emu-domain \
    --output results/blend_lookup_both_grids_sw100_allpairs_c40-139.feather || exit 1

echo
echo "############ constgold m"
python -u scripts/eval_m_with_blendflow.py \
    --flow-lookup results/blend_lookup_both_grids_sw100_allpairs_c40-139.feather \
    --emu-lookup  results/blend_lookup_both_grids_sw100_allpairs_c40-139.feather \
    --flow-col R_blend_flow --emu-col R_blend_emu --in-domain-only || exit 1

echo "BFCG2_DONE"
