#!/bin/bash
#SBATCH --job-name=hspair_prep
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather
MANIFEST=$ROOT/results/halfshear_noiseless_pair_toy_manifest_v22_n20000.feather
AUDIT=$ROOT/results/halfshear_noiseless_pair_toy_manifest_v22_n20000.json
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
for output in "$MANIFEST" "$AUDIT"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done
"$PY" -u scripts/prepare_halfshear_noiseless_pair_toys.py \
  --catalogue "$CAT" --sample-size 20000 --seed 20260813 \
  --case-min 40 --case-max 199 --catalogue-shear 0.2 \
  --tag lsst_r_extnbr_v22 --expected-eligible 37852393 \
  --output-feather "$MANIFEST" --output-json "$AUDIT"
test -s "$MANIFEST"
test -s "$AUDIT"
echo HALFSHEAR_NOISELESS_PAIR_TOY_PREP_JOB_DONE
date
