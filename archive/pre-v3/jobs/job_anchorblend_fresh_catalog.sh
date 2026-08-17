#!/bin/bash
#SBATCH --job-name=abfr_cat
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/abfr_cat_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abfr_cat_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
CFG=$ROOT/configs/fs2_lsst_r_anchorblend_g005_c300-399.yaml
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c300-399
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
test ! -e "$BASE" || { echo "REFUSING existing $BASE"; exit 1; }
cd /home/z/Zekang.Zhang/blendemu/scripts
python -u run_pipeline.py --config "$CFG" --steps 1
cd "$ROOT"
python -u scripts/prepare_anchorblend_catalogues.py \
  --base "$BASE" --cases $(seq 300 399) --g 0.05 --min-separation 20
for case in $(seq 300 399); do
  test -s "$BASE/gals${case}_0.05.feather"
  test -s "$BASE/gals${case}_-0.05.feather"
  test -s "$BASE/anchors_case${case}.feather"
done
echo ANCHORBLEND_FRESH_CATALOG_DONE
date
