#!/bin/bash
#SBATCH --job-name=v22_shdecomp
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22_shdecomp_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22_shdecomp_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
PAIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_constgold_pair_features_c40-139.feather
HALF=/project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v22_c40-199/halfshear_selfresp_v22_c40-199.feather
OUT_JSON=$ROOT/results/v22_shared_carrier_exactkey_decomposition_c40-139.json
OUT_MD=$ROOT/results/v22_shared_carrier_exactkey_decomposition_c40-139.md
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for output in "$OUT_JSON" "$OUT_MD"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
cd "$ROOT"
python -u scripts/decompose_v22_shared_carrier.py \
  --constgold-gap results/v22_constgold_gap_features_c40-139.feather \
  --constgold-pairs "$PAIR" --half-selfresp "$HALF" \
  --output-json "$OUT_JSON" --output-md "$OUT_MD"
echo V22_SHARED_CARRIER_DECOMPOSITION_JOB_DONE
date
