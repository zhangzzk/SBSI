#!/usr/bin/env bash
#SBATCH --job-name=anchorblend_cgedges
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

# The scene population read on ConstGold's own R_blend_abs quantile edges, so
# the two cohorts are compared bin for bin rather than on their means.
EDGES=$("$PY" -c "
import json
edges = json.load(open('$V2/constgold_flow_quiet_v1/result.json'))['binning']['edges']
print(' '.join('--edge %.17g' % e for e in edges if e is not None))
")

echo "############ matched, ConstGold edges ############"
# shellcheck disable=SC2086
"$PY" "$SBSI/scripts/diagnose_anchorblend_response_by_crowding.py" \
  --parts "$OUT/parts" --branch matched --bin-column R_scene_abs $EDGES \
  --output "$OUT/crowding_matched_constgold_edges.json"

echo "############ matched, ConstGold top bin alone ############"
"$PY" "$SBSI/scripts/diagnose_anchorblend_response_by_crowding.py" \
  --parts "$OUT/parts" --branch matched --bin-column R_scene_abs \
  --edge 0.4915618896484374 --edge 0.8674044509728747 \
  --output "$OUT/crowding_matched_constgold_tail.json"

echo "############ how the two cohorts sit on the same axis ############"
"$PY" - <<'PYEOF'
import json
import numpy as np
import pandas as pd

V2 = "/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2"
edges = [e for e in json.load(
    open(f"{V2}/constgold_flow_quiet_v1/result.json"))["binning"]["edges"] if e is not None]

scene = pd.concat(
    (pd.read_feather(p, columns=["R_scene_abs", "R_scene", "weight", "matched"])
     for p in sorted(__import__("pathlib").Path(
         f"{V2}/anchorblend_g002_primaries_v1/parts").glob("case*.feather"))),
    ignore_index=True)
scene = scene[scene["matched"]]
constgold = pd.read_feather(
    f"{V2}/constgold_flow_quiet_v1/rblend_v2_baseline_abs_cases40_89.feather",
    columns=["R_blend_abs", "R_blend"])

rows = []
for name, values, blend, weight in (
    ("scene", scene["R_scene_abs"].to_numpy(), scene["R_scene"].to_numpy(),
     scene["weight"].to_numpy()),
    ("constgold", constgold["R_blend_abs"].to_numpy(), constgold["R_blend"].to_numpy(),
     np.ones(len(constgold))),
):
    which = np.searchsorted(edges, values, side="right")
    share_n = np.bincount(which, weight, len(edges) + 1)
    share_b = np.bincount(which, weight * np.abs(blend), len(edges) + 1)
    rows.append(pd.DataFrame({
        "cohort": name,
        "bin": [f"bin{i:02d}" for i in range(len(edges) + 1)],
        "objects_percent": 100 * share_n / share_n.sum(),
        "blending_percent": 100 * share_b / share_b.sum(),
    }))
table = pd.concat(rows).pivot(index="bin", columns="cohort",
                              values=["objects_percent", "blending_percent"])
print(table.round(3).to_string())
print("\nweighted mean |R_blend| : scene %.5f  constgold %.5f" % (
    np.average(scene["R_scene_abs"], weights=scene["weight"]),
    constgold["R_blend_abs"].mean()))
print("weighted mean  R_blend  : scene %.5f  constgold %.5f" % (
    np.average(scene["R_scene"], weights=scene["weight"]),
    constgold["R_blend"].mean()))
PYEOF
