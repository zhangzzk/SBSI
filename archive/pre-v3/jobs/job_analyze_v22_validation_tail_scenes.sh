#!/bin/bash
#SBATCH --job-name=v22valtail
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather
META=/home/z/Zekang.Zhang/blendemu/models/emulator_metadata_lsst_r_extnbr_v22.json
REFERENCE=$ROOT/results/v22_validation_cumulative_response_mag_fluxratio_c40-199.json
SCENES=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_validation_tail_scenes_c40-199.feather
OUT=$ROOT/results/v22_validation_tail_scene_residual_c40-199
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for input in "$CAT" "$META" "$REFERENCE"; do
  test -s "$input" || { echo "MISSING $input"; exit 1; }
done
for output in "$SCENES" "$OUT.csv" "$OUT.cases.csv" "$OUT.json" "$OUT.md" "$OUT.pdf" "$OUT.png"; do
  test ! -e "$output" || { echo "REFUSING existing $output"; exit 1; }
done

cd "$ROOT"
python -u scripts/analyze_v22_validation_tail_scenes.py \
  --catalogue "$CAT" \
  --model-metadata "$META" \
  --reference-validation-json "$REFERENCE" \
  --tag lsst_r_extnbr_v22 \
  --case-min 40 --case-max 199 --shear 0.2 \
  --score-chunk 1000000 \
  --scene-catalogue "$SCENES" \
  --output-prefix "$OUT"
echo V22_VALIDATION_TAIL_SCENES_DONE
