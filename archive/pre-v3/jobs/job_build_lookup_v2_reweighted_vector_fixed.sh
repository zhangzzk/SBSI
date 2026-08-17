#!/bin/bash
#SBATCH --job-name=lk_v2_rwvfix
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/lk_v2_rwvfix_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/lk_v2_rwvfix_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
TRAIN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_reweighted_vector_fixed_v1
MODEL=$TRAIN/weighted_model.json
OUT=results/blend_lookup_v2_reweighted_vector_fixed_c40-139.feather

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
cd "$ROOT"

test -s "$TRAIN/summary.json"
test -s "$MODEL"
"$PY" - "$TRAIN/summary.json" "$MODEL" <<'PY'
import hashlib, json, sys
summary, model = sys.argv[1:]
expected = json.load(open(summary))["artifacts"]["weighted_model_sha256"]
actual = hashlib.sha256(open(model, "rb").read()).hexdigest()
if actual != expected:
    raise SystemExit(f"weighted model hash mismatch: {actual} != {expected}")
PY
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
"$PY" -u scripts/build_blend_lookup.py \
  --cases $(seq 40 139) \
  --tag lsst_r_extnbr_indom_tuned \
  --reg-file "$MODEL" \
  --output "$OUT"
test -s "$OUT"
echo V2_REWEIGHTED_VECTOR_FIXED_LOOKUP_DONE
