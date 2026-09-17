#!/usr/bin/env bash
#SBATCH --job-name=anchorblend_crowding
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

OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/anchorblend_g002_primaries_v1
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
SBSI=/home/z/Zekang.Zhang/SBSI
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

run () {
  echo "############ $* ############"
  "$PY" "$SBSI/scripts/diagnose_anchorblend_response_by_crowding.py" \
    --parts "$OUT/parts" "$@"
}

run --branch matched   --bin-column R_scene_abs --bins 8  --output "$OUT/crowding_matched_abs8.json"
run --branch matched   --bin-column R_scene_abs --bins 12 --output "$OUT/crowding_matched_abs12.json"
run --branch matched   --bin-column R_scene     --bins 8  --output "$OUT/crowding_matched_signed8.json"
run --branch unmatched --bin-column R_scene_abs --bins 8  --output "$OUT/crowding_unmatched_abs8.json"
