#!/bin/bash
#SBATCH --job-name=ab_bias_cg
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/py31
PAIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_constgold_pair_features_biasfull_c40-139.feather
PREFIX=${CG_BIAS_PREFIX:-$ROOT/results/anchor_bias_emulator_transfer_constgold_v22_c40-139}
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for input in \
  "$PAIR" \
  "$ROOT/results/v22_constgold_gap_features_c40-139.feather" \
  "$ROOT/results/anchorblend_g002_bias_features_v22_c400-899.feather" \
  "$ROOT/results/anchorblend_g002_bias_emulator_v22_c400-899_final.joblib" \
  "$ROOT/results/anchorblend_g002_bias_emulator_v22_c400-899_final.json" \
  /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps/ablate_s2c_lt500_v22_perobj_s501.feather; do
  test -s "$input" || { echo "MISSING $input"; exit 1; }
done
for suffix in .json .md _cases.csv _deciles.csv _support.csv; do
  test ! -e "${PREFIX}${suffix}" || {
    echo "REFUSING existing ${PREFIX}${suffix}"; exit 1; }
done
cd "$ROOT"

python -u scripts/apply_anchor_bias_emulator_constgold.py \
  --pair-features "$PAIR" \
  --constgold-gap results/v22_constgold_gap_features_c40-139.feather \
  --anchor-features results/anchorblend_g002_bias_features_v22_c400-899.feather \
  --bias-emulator results/anchorblend_g002_bias_emulator_v22_c400-899_final.joblib \
  --bias-emulator-json results/anchorblend_g002_bias_emulator_v22_c400-899_final.json \
  --dump-glob '/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps/ablate_s2c_lt500_v22_perobj_s*.feather' \
  --output-prefix "$PREFIX"
echo ANCHOR_BIAS_EMULATOR_CONSTGOLD_JOB_DONE
date
