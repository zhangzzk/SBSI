#!/bin/bash
#SBATCH --job-name=lk_v22_phys2_pilot
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/lk_v22_phys2_pilot_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
OUT="$ROOT/results/blend_lookup_v22_phys2_c40-49.feather"
MODEL=/home/z/Zekang.Zhang/blendemu/models/regression_model_lsst_r_extnbr_v22_phys2.json
META=/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_v22_phys2.json
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant

export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export XGB_DEVICE=cpu

cd "$ROOT"
[ -f "$MODEL" ] || { echo "MISSING model: $MODEL"; exit 1; }
[ -f "$META" ] || { echo "MISSING metadata: $META"; exit 1; }
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
for case_id in $(seq 40 49); do
  input="$BASE/case${case_id}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather"
  [ -f "$input" ] || { echo "MISSING input: $input"; exit 1; }
done

echo "### V2.2 PHYS2 CONSTGOLD PILOT job=$SLURM_JOB_ID cases=40-49 ###"
date
"$PY" -u scripts/build_blend_lookup.py \
  --cases $(seq 40 49) \
  --tag lsst_r_extnbr_v22_phys2 \
  --output "$OUT"
[ -f "$OUT" ] || { echo "LOOKUP_FAILED: no output"; exit 1; }
echo "LOOKUP_DONE: $OUT"
date
