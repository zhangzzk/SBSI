#!/bin/bash
#SBATCH --job-name=ab_overlap
#SBATCH --time=00:30:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_overlap_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
cd "$ROOT"

OUT=results/anchorblend_training_overlap_audit_v2.json
[ ! -e "$OUT" ] || { echo "REFUSING existing $OUT"; exit 1; }

"$PY" -u scripts/audit_anchorblend_training_overlap.py \
  --response-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --anchor-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005 \
  --cases 0 40 99 \
  --output results/anchorblend_training_overlap_audit_v2_0-99.json

"$PY" -u scripts/audit_anchorblend_training_overlap.py \
  --response-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --anchor-root /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext \
  --cases 100 150 199 \
  --output results/anchorblend_training_overlap_audit_v2_100-199.json

"$PY" - <<'PY'
import json
parts = [
    json.load(open('results/anchorblend_training_overlap_audit_v2_0-99.json')),
    json.load(open('results/anchorblend_training_overlap_audit_v2_100-199.json')),
]
payload = {
    'audited_cases': [row for part in parts for row in part['cases']],
    'all_audited_cases_exact': all(part['all_cases_exact'] for part in parts),
    'all_audited_cases_exact_intrinsic_prefix': all(
        part['all_cases_exact_intrinsic_prefix'] for part in parts
    ),
    'interpretation': (
        'The anchor field has one quarter as many objects, so whole-table equality is not '
        'the relevant check. exact_intrinsic_prefix tests whether it reuses the first '
        'quarter of the response field galaxy draws while rerandomizing geometry/shear. '
        'Cases 0-39 are excluded from V2.2 training; cases 200-299 have no corresponding '
        'response-emulator fields.'
    ),
}
json.dump(payload, open('results/anchorblend_training_overlap_audit_v2.json', 'w'),
          indent=2, sort_keys=True, allow_nan=False)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo ANCHORBLEND_TRAINING_OVERLAP_JOB_DONE; date
