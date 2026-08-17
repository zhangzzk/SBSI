#!/bin/bash
#SBATCH --job-name=abg002_cat
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abg002_cat_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abg002_cat_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
CFG=$ROOT/configs/fs2_lsst_r_anchorblend_g002_c400-499.yaml
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c400-499
REFERENCE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c400-599
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
test ! -e "$BASE" || { echo "REFUSING existing $BASE"; exit 1; }
cd /home/z/Zekang.Zhang/blendemu/scripts
python -u run_pipeline.py --config "$CFG" --steps 1
cd "$ROOT"
python -u scripts/prepare_anchorblend_catalogues.py \
  --base "$BASE" --cases $(seq 400 499) --g 0.02 --min-separation 20
python - <<'PY'
import os
import pandas as pd

new = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c400-499"
old = "/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c400-599"
columns = ["index", "cata_idx", "RA", "DEC", "r", "Re"]
for case in range(400, 500):
    left = pd.read_feather(os.path.join(new, f"anchors_case{case}.feather"))[columns]
    right = pd.read_feather(os.path.join(old, f"anchors_case{case}.feather"))[columns]
    if not left.equals(right):
        raise RuntimeError(f"case {case}: g=.02 and g=.05 anchor manifests differ")
print("paired manifests are exactly identical for cases 400--499")
PY
echo ANCHORBLEND_G002_CATALOG_DONE
date
