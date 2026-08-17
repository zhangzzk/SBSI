#!/bin/bash
#SBATCH --job-name=hs_det_proj
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_det_proj_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_det_proj_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
cd "$ROOT"
"$PY" -u scripts/analyze_halfshear_detection_projected.py \
  --input /project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_coherence_corrected_c0-39.feather \
  --output results/halfshear_detection_projected_v22_c0-39.json
echo HALFSHEAR_DETECTION_PROJECTED_JOB_DONE
date
