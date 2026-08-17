#!/bin/bash
#SBATCH --job-name=emu_decompose
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_decompose_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/emu_decompose_%j.err
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
"$PY" -u scripts/analyze_weighted_correction_decomposition.py \
  --mean-json results/emulator_global_mean_v22_rpowposa0065_c40-199.json \
  --halfshear-baseline /project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_coherence_corrected_c0-39.feather \
  --halfshear-candidate results/halfshear_vector_closure_lsst_r_extnbr_v22_rpowposa0065_c0-39.feather \
  --independent-baseline results/anchorblend_independent_local10_response_v22_c200-299.feather \
  --independent-candidate /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_independent_lsst_r_extnbr_v22_rpowposa0065_c200-299.feather \
  --coherent-old-root /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpow_positive_c200-399 \
  --coherent-fresh-root /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpow_positive_c400-599 \
  --tag lsst_r_extnbr_v22_rpowposa0065 \
  --output results/weighted_correction_decomposition_v22_rpowposa0065.json
