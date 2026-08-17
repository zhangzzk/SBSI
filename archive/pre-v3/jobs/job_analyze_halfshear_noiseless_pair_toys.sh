#!/bin/bash
#SBATCH --job-name=hspair_plot
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
MANIFEST=$ROOT/results/halfshear_noiseless_pair_toy_manifest_v22_n20000.feather
INDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/halfshear_noiseless_pair_toy_v22_n20000_az8
PREFIX=${PREFIX:-$ROOT/results/halfshear_noiseless_pair_toy_rblend_v22_n20000_az8_central99}
TAIL=${TAIL:-0.005}
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
for suffix in feather rotations.feather json md pdf png; do
  test ! -e "$PREFIX.$suffix" || { echo "REFUSING existing $PREFIX.$suffix"; exit 1; }
done
"$PY" -u scripts/analyze_halfshear_noiseless_pair_toys.py \
  --manifest "$MANIFEST" --input-dir "$INDIR" --n-shards 40 \
  --output-prefix "$PREFIX" --bins 160 --tail-probability "$TAIL" \
  --min-success-fraction 0.98
for suffix in feather rotations.feather json md pdf png; do test -s "$PREFIX.$suffix"; done
echo HALFSHEAR_NOISELESS_PAIR_TOY_ANALYSIS_JOB_DONE
date
