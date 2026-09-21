#!/bin/bash
#SBATCH --job-name=flowself
#SBATCH --partition=cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%A_%a.out

set -euo pipefail
export NO_EXPANDABLE_SEGMENTS=1
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
V2=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2
OUT="$V2/flow_self_response_v1"
mkdir -p "$OUT/logs"

#  Which half of the flow's case split this task reads.  The validation cases
#  are the flow's held-out sample; the trained cases are taken as the forty
#  lowest-numbered training cases outside the ones already predicted, so the
#  selection is fixed by the split, not by any result.
SPLIT="$1"
CASES=$("$PY" - "$SPLIT" <<'PYEOF'
import json, sys
split = sys.argv[1]
manifest = json.load(open("/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/domain/manifest.json"))
train = sorted(int(c) for c in manifest["split"]["train_cases"])
validation = sorted(int(c) for c in manifest["split"]["validation_cases"])
cases = validation if split == "validation" else [c for c in train if c >= 20][:40]
print(" ".join(f"--case {c}" for c in cases))
PYEOF
)

echo "split=$SPLIT"
echo "cases: $CASES"
# shellcheck disable=SC2086
srun "$PY" scripts/build_flow_self_response_predictions.py \
  --domain-root "$V2/domain" \
  --flow "$V2/flow/paired/selected.pt" \
  --output "$OUT/R_self_model_${SPLIT}.feather" \
  --draws 64 \
  --batch-size 1024 \
  --device cuda \
  $CASES
