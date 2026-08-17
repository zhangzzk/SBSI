#!/bin/bash
#SBATCH --job-name=abv2c_smoke
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abv2c_smoke_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abv2c_smoke_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BE=/home/z/Zekang.Zhang/blendemu
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
CFG=$ROOT/configs/fs2_lsst_r_anchorblend_g002_c400-499.yaml
BASE=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/anchor_v2_complement_smoke_case400
REFERENCE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c400-499

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:$BE:${PYTHONPATH:-}"
cd "$BE/scripts"
[ ! -e "$BASE" ] || { echo "REFUSING existing $BASE"; exit 1; }
python -u run_pipeline.py --config "$CFG" --steps 1 --n-cases 1 \
  --case-offset 400 --output-path "$BASE"
cd "$ROOT"
python -u scripts/prepare_anchorblend_v2_complement.py \
  --base "$BASE" --reference-base "$REFERENCE" --cases 400 \
  --g 0.02 --min-separation 20
test -s "$BASE/anchors_case400.feather"
test -s "$BASE/anchor_population_case400.json"
echo ANCHOR_V2_COMPLEMENT_SMOKE_DONE
