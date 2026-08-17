#!/bin/bash
#SBATCH --job-name=v22dc_cat
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22dc_cat_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22dc_cat_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BE=/home/z/Zekang.Zhang/blendemu
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
CFG=$ROOT/configs/fs2_lsst_r_v22_matched_decomp_pilot.yaml
PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_v22decomp_pilot
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_matched_decomp_pilot
MODES=(total self neighbour pair0 pair1)
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:$BE:${PYTHONPATH:-}"

for mode in "${MODES[@]}"; do
  base="${PREFIX}_${mode}"
  if [ -e "$base" ]; then
    echo "REFUSING: pilot base already exists: $base"
    exit 1
  fi
done
if [ -e "$MANIFEST" ]; then
  echo "REFUSING: pilot manifest already exists: $MANIFEST"
  exit 1
fi

cd "$BE/scripts"
for mode in "${MODES[@]}"; do
  "$PY" -u run_pipeline.py --config "$CFG" --steps 1 \
    --output-path "${PREFIX}_${mode}"
done

cd "$ROOT"
"$PY" -u scripts/prepare_v22_matched_decomposition.py \
  --base "total=${PREFIX}_total" \
  --base "self=${PREFIX}_self" \
  --base "neighbour=${PREFIX}_neighbour" \
  --base "pair0=${PREFIX}_pair0" \
  --base "pair1=${PREFIX}_pair1" \
  --cases 300 301 302 303 --g 0.05 --min-separation 20.01 \
  --tag lsst_r_extnbr_v22 --random-seed 25876 --manifest-dir "$MANIFEST"
echo V22_MATCHED_DECOMP_PILOT_CATALOG_DONE

