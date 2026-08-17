#!/bin/bash
#SBATCH --job-name=hspair_toy
#SBATCH --array=0-39%40
#SBATCH --time=03:00:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
MANIFEST=$ROOT/results/halfshear_noiseless_pair_toy_manifest_v22_n20000.feather
OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/halfshear_noiseless_pair_toy_v22_n20000_az8
SHARD=${SLURM_ARRAY_TASK_ID:?array task required}
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
mkdir -p "$OUTDIR"
PAIR=$(printf '%s/pairs_shard%03d.feather' "$OUTDIR" "$SHARD")
ROTATIONS=$(printf '%s/rotations_shard%03d.feather' "$OUTDIR" "$SHARD")
AUDIT=$(printf '%s/audit_shard%03d.json' "$OUTDIR" "$SHARD")
for output in "$PAIR" "$ROTATIONS" "$AUDIT"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
cd "$ROOT"
"$PY" -u scripts/run_halfshear_noiseless_pair_toy_shard.py \
  --manifest "$MANIFEST" --shard-index "$SHARD" --n-shards 40 \
  --g 0.05 --stamp 112 --fit-seed 42 --n-azimuths 8 \
  --output-feather "$PAIR" --output-rotations-feather "$ROTATIONS" \
  --output-json "$AUDIT"
test -s "$PAIR"
test -s "$ROTATIONS"
test -s "$AUDIT"
echo HALFSHEAR_NOISELESS_PAIR_TOY_SHARD_JOB_DONE
date
