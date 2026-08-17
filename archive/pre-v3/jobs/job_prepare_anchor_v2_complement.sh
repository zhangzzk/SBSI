#!/bin/bash
#SBATCH --job-name=abv2c_prep
#SBATCH --time=01:30:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --array=0-4%2
#SBATCH --output=/home/z/Zekang.Zhang/logs/abv2c_prep_%A_%a.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abv2c_prep_%A_%a.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BE=/home/z/Zekang.Zhang/blendemu
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
START=$((400 + 100 * SLURM_ARRAY_TASK_ID))
STOP=$((START + 99))
CFG=$ROOT/configs/fs2_lsst_r_anchorblend_g002_c${START}-${STOP}.yaml
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_v2complement_c${START}-${STOP}
REFERENCE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c${START}-${STOP}

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:$BE:${PYTHONPATH:-}"
[ ! -e "$BASE" ] || { echo "REFUSING existing $BASE"; exit 1; }
cd "$BE/scripts"
python -u run_pipeline.py --config "$CFG" --steps 1 --output-path "$BASE"
cd "$ROOT"
python -u scripts/prepare_anchorblend_v2_complement.py \
  --base "$BASE" --reference-base "$REFERENCE" \
  --cases $(seq "$START" "$STOP") --g 0.02 --min-separation 20
for case in $(seq "$START" "$STOP"); do
  test -s "$BASE/anchors_case${case}.feather"
  test -s "$BASE/anchor_population_case${case}.json"
done
echo "ANCHOR_V2_COMPLEMENT_PREP_DONE cases=$START-$STOP"
