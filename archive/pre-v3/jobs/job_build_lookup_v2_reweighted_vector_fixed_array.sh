#!/bin/bash
#SBATCH --job-name=lk_v2_rwv_arr
#SBATCH --array=0-9
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/lk_v2_rwv_arr_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/lk_v2_rwv_arr_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
TRAIN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1
MODEL=$TRAIN/weighted_model.json
PARTS=$TRAIN/constgold_lookup_parts
START=$((40 + 10 * SLURM_ARRAY_TASK_ID))
END=$((START + 9))
OUT=$PARTS/lookup_c$(printf '%03d' "$START")-$(printf '%03d' "$END").feather

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
cd "$ROOT"

mkdir -p "$PARTS"
test -s "$TRAIN/summary.json"
test -s "$MODEL"
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
"$PY" -u scripts/build_blend_lookup.py \
  --cases $(seq "$START" "$END") \
  --tag lsst_r_extnbr_indom_tuned \
  --reg-file "$MODEL" \
  --output "$OUT"
test -s "$OUT"
echo "V2_REWEIGHTED_VECTOR_LOOKUP_SHARD_DONE cases=$START-$END"
