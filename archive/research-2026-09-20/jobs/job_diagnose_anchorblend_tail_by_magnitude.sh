#!/usr/bin/env bash
#SBATCH --job-name=anchorblend_tailmag
#SBATCH --partition=cluster
#SBATCH --qos=normal
#SBATCH --account=ls-gruen
#SBATCH --constraint=x86-64-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

V2=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2
OUT=$V2/anchorblend_g002_primaries_v1
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
SBSI=/home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# ConstGold's two top edges, so the tail is the same slice of the blending axis
# in every band; the bands say whether the tail failure moves with brightness.
for band in "0 24" "24 25" "25 26"; do
  set -- $band
  echo "############ primary r in [$1, $2) ############"
  "$PY" "$SBSI/scripts/diagnose_anchorblend_response_by_crowding.py" \
    --parts "$OUT/parts" --branch matched --bin-column R_scene_abs \
    --edge 0.4915618896484374 --edge 0.8674044509728747 \
    --primary-magnitude-min "$1" --primary-magnitude-max "$2" \
    --output "$OUT/crowding_matched_tail_r${1}_${2}.json"
done

echo "############ magnitude composition of the tail, both cohorts ############"
"$PY" - <<'PYEOF'
import json
from pathlib import Path
import numpy as np
import pandas as pd

V2 = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2"
cut = 0.8674044509728747
bands = [-np.inf, 23, 24, 25, 26, np.inf]
names = ["r<23", "23-24", "24-25", "25-26", "r>=26"]

scene = pd.concat(
    (pd.read_feather(p, columns=["R_scene_abs", "r_input_p", "weight", "matched"])
     for p in sorted(Path(f"{V2}/anchorblend_g002_primaries_v1/parts").glob("case*.feather"))),
    ignore_index=True)
scene = scene[scene["matched"]]
rows = {}
for label, values, mag, weight in (
    ("scene", scene["R_scene_abs"].to_numpy(), scene["r_input_p"].to_numpy(),
     scene["weight"].to_numpy()),
):
    for where, selection in (("cohort", np.ones(len(values), bool)), ("tail", values > cut)):
        band = np.searchsorted(bands[1:-1], mag[selection], side="right")
        counts = np.bincount(band, weight[selection], len(names))
        rows[f"{label}_{where}"] = 100 * counts / counts.sum()
print(pd.DataFrame(rows, index=names).round(2).to_string())
PYEOF
