#!/bin/bash
#SBATCH --job-name=v22cg_pairphys
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22cg_pairphys_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22cg_pairphys_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_constgold_pair_features_c40-139.feather
SUMMARY=$ROOT/results/v22_constgold_pair_features_c40-139.json
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for output in "$OUT" "$SUMMARY"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
cd "$ROOT"
python -u scripts/build_v22_constgold_pair_features.py \
  --cases $(seq 40 139) \
  --tag lsst_r_extnbr_v22 \
  --reference-lookup results/blend_lookup_v22_c40-139.feather \
  --output "$OUT" --summary-json "$SUMMARY"
echo V22_CONSTGOLD_PAIR_FEATURES_JOB_DONE
date
