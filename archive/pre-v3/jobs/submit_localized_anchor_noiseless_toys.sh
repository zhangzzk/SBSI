#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SBATCH=/opt/slurm/bin/sbatch
MANIFEST=$ROOT/results/localized_anchor_noiseless_toy_manifest_c700-899.feather
MANIFEST_JSON=$ROOT/results/localized_anchor_noiseless_toy_manifest_c700-899.json
SHARDS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/localized_anchor_noiseless_toy_v22_c700-899
OUT=$ROOT/results/localized_anchor_noiseless_toy_v22_c700-899
cd "$ROOT"

for output in "$OUT.feather" "${OUT}_pairs.feather" "$OUT.json" "$OUT.md"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
test ! -e "$SHARDS" || { echo "REFUSING existing shard directory $SHARDS"; exit 1; }

dependency=()
if test -s "$MANIFEST" && test -s "$MANIFEST_JSON"; then
  echo "reusing frozen manifest $MANIFEST"
else
  test ! -e "$MANIFEST"
  test ! -e "$MANIFEST_JSON"
  prep=$($SBATCH --parsable jobs/job_prepare_localized_anchor_noiseless_toys.sh)
  dependency=(--dependency=afterok:"$prep")
  echo "prepare=$prep"
fi
array=$($SBATCH --parsable "${dependency[@]}" jobs/job_run_localized_anchor_noiseless_toy.sh)
analysis=$($SBATCH --parsable --dependency=afterok:"$array" jobs/job_analyze_localized_anchor_noiseless_toys.sh)
printf '%s\n' "array=$array" "analysis=$analysis"
