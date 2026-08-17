#!/bin/bash
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
SBATCH=/opt/slurm/bin/sbatch
MANIFEST=$ROOT/results/halfshear_noiseless_pair_toy_manifest_v22_n20000.feather
AUDIT=$ROOT/results/halfshear_noiseless_pair_toy_manifest_v22_n20000.json
SHARDS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/halfshear_noiseless_pair_toy_v22_n20000_az8
PREFIX=$ROOT/results/halfshear_noiseless_pair_toy_rblend_v22_n20000_az8_central99
cd "$ROOT"

test ! -e "$SHARDS" || { echo "REFUSING existing shard directory $SHARDS"; exit 1; }
for suffix in feather rotations.feather json md pdf png; do
  test ! -e "$PREFIX.$suffix" || { echo "REFUSING existing $PREFIX.$suffix"; exit 1; }
done

if test -s "$MANIFEST" && test -s "$AUDIT"; then
  jq -e '
    .sample_size == 20000 and
    .sample_seed == 20260813 and
    .n_eligible == 37852393 and
    .tag == "lsst_r_extnbr_v22"
  ' "$AUDIT" >/dev/null
  prepare=reused
  array=$($SBATCH --parsable jobs/job_run_halfshear_noiseless_pair_toy.sh)
elif test -e "$MANIFEST" || test -e "$AUDIT"; then
  echo "REFUSING incomplete manifest/audit pair" >&2
  exit 1
else
  prepare=$($SBATCH --parsable jobs/job_prepare_halfshear_noiseless_pair_toys.sh)
  array=$($SBATCH --parsable --dependency=afterok:"$prepare" jobs/job_run_halfshear_noiseless_pair_toy.sh)
fi
analysis=$($SBATCH --parsable --dependency=afterok:"$array" jobs/job_analyze_halfshear_noiseless_pair_toys.sh)
printf '%s\n' "prepare=$prepare" "array=$array" "analysis=$analysis"
