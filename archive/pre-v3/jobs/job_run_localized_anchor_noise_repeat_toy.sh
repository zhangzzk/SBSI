#!/bin/bash
#SBATCH --job-name=abnr_scene
#SBATCH --array=0-19%20
#SBATCH --time=02:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
SCENE_ID=${SLURM_ARRAY_TASK_ID:?array task required}
MANIFEST=$ROOT/results/localized_anchor_noise_repeat_toy_manifest_v22_typical20.feather
OUTDIR=${TOY_OUTPUT_DIR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/localized_anchor_noise_repeat_toy_v22_typical20_r400}
NREAL=${TOY_NREAL:-400}
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
mkdir -p "$OUTDIR"
DRAW=$OUTDIR/draws_scene$(printf '%02d' "$SCENE_ID").feather
PAIR=$OUTDIR/pairs_scene$(printf '%02d' "$SCENE_ID").feather
AUDIT=$OUTDIR/audit_scene$(printf '%02d' "$SCENE_ID").json
for output in "$DRAW" "$PAIR" "$AUDIT"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
cd "$ROOT"
"$PY" -u scripts/run_localized_anchor_noise_repeat_toy_scene.py \
  --manifest "$MANIFEST" \
  --pair-design \
    results/anchorblend_g002_pairs_renderer_v22_c700-799.json \
    results/anchorblend_g002_pairs_renderer_v22_c800-899.json \
  --scene-id "$SCENE_ID" --nreal "$NREAL" --g 0.02 --stamp 48 \
  --pixel-rms 0.312 \
  --output-draws "$DRAW" --output-pairs "$PAIR" --output-json "$AUDIT"
test -s "$DRAW"
test -s "$PAIR"
test -s "$AUDIT"
echo LOCALIZED_ANCHOR_NOISE_REPEAT_SCENE_JOB_DONE
date
