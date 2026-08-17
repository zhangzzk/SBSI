#!/bin/bash
#SBATCH --job-name=abg002x_cat
#SBATCH --time=01:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/%x_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/%x_%j.err
set -euo pipefail

# Extension of the g=0.02 coherent-anchor block beyond the original cases
# 400--499.  Parameterised by AB_START / AB_STOP, which the submit wrapper
# exports; every other setting mirrors jobs/job_anchorblend_g002_catalog.sh.
: "${AB_START:?AB_START not set}"
: "${AB_STOP:?AB_STOP not set}"

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
CFG=$ROOT/configs/fs2_lsst_r_anchorblend_g002_c${AB_START}-${AB_STOP}.yaml
BASE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g002_c${AB_START}-${AB_STOP}
REFERENCE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c400-599
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
test -f "$CFG" || { echo "MISSING config $CFG"; exit 1; }
test ! -e "$BASE" || { echo "REFUSING existing $BASE"; exit 1; }
cd /home/z/Zekang.Zhang/blendemu/scripts
python -u run_pipeline.py --config "$CFG" --steps 1
cd "$ROOT"
python -u scripts/prepare_anchorblend_catalogues.py \
  --base "$BASE" --cases $(seq "$AB_START" "$AB_STOP") --g 0.02 --min-separation 20

# The g=0.05 anchor tree only covers cases 400--599, so the paired-manifest
# identity audit can only run on the overlap.  Where it can run it must pass;
# beyond case 599 there is no partner to compare against and the extension is
# deliberately unpaired (the noise-limited quantity is the g=0.02 leg alone).
AB_START="$AB_START" AB_STOP="$AB_STOP" REFERENCE="$REFERENCE" BASE="$BASE" python - <<'PY'
import os

import pandas as pd

start = int(os.environ["AB_START"])
stop = int(os.environ["AB_STOP"])
new = os.environ["BASE"]
old = os.environ["REFERENCE"]
columns = ["index", "cata_idx", "RA", "DEC", "r", "Re"]
overlap = [case for case in range(start, stop + 1)
           if os.path.exists(os.path.join(old, f"anchors_case{case}.feather"))]
for case in overlap:
    left = pd.read_feather(os.path.join(new, f"anchors_case{case}.feather"))[columns]
    right = pd.read_feather(os.path.join(old, f"anchors_case{case}.feather"))[columns]
    if not left.equals(right):
        raise RuntimeError(f"case {case}: g=.02 and g=.05 anchor manifests differ")
if overlap:
    print(f"paired manifests exactly identical for {len(overlap)} overlap cases "
          f"{overlap[0]}--{overlap[-1]}")
else:
    print(f"no g=0.05 partner for cases {start}--{stop}: unpaired block, audit skipped")
PY
echo "ANCHORBLEND_G002_EXT_CATALOG_DONE cases=${AB_START}-${AB_STOP}"
date
