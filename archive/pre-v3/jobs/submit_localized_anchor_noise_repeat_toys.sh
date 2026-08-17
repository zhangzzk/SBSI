#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SBATCH=/opt/slurm/bin/sbatch
MANIFEST=$ROOT/results/localized_anchor_noise_repeat_toy_manifest_v22_typical20
SHARDS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/localized_anchor_noise_repeat_toy_v22_typical20_r400
OUT=$ROOT/results/localized_anchor_noise_repeat_toy_v22_typical20_r400
cd "$ROOT"

for output in "$MANIFEST.feather" "$MANIFEST.json" "$OUT.feather" \
  "${OUT}_pairs.feather" "$OUT.json" "$OUT.md"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
test ! -e "$SHARDS" || { echo "REFUSING existing shard directory $SHARDS"; exit 1; }

prep=$($SBATCH --parsable jobs/job_prepare_localized_anchor_noise_repeat_toys.sh)
array=$($SBATCH --parsable --dependency=afterok:"$prep" \
  jobs/job_run_localized_anchor_noise_repeat_toy.sh)
analysis=$($SBATCH --parsable --dependency=afterok:"$array" \
  jobs/job_analyze_localized_anchor_noise_repeat_toys.sh)
printf '%s\n' "prepare=$prep" "array=$array" "analysis=$analysis"
