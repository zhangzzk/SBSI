#!/bin/bash
#SBATCH --job-name=lk_indom_v3b
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/lk_indom_v3b_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/lk_indom_v3b_%j.err

# Evaluate the fixed old-domain bridge emulator on the exact constgold case/input
# population used by the historical selection table.  This reads only truth fields
# and positions; no constgold measurement is used for fitting.
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}
export XGB_DEVICE=cpu
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
mkdir -p results
"$PY" -u scripts/check_response_weighted_provenance.py \
  --tag lsst_r_extnbr_indom_tuned_rpowposa0065_all200 \
  --source lsst_r_extnbr_indom_tuned --alpha 0.065 --cap 50 \
  --mode positive_square --trees 271 --minimum-training-case 0
"$PY" -u scripts/build_blend_lookup.py --cases $(seq 40 139) \
  --tag lsst_r_extnbr_indom_tuned_rpowposa0065_all200 \
  --output results/blend_lookup_indom_v3bridge_c40-139.feather
test -s results/blend_lookup_indom_v3bridge_c40-139.feather
echo LK_INDOM_V3BRIDGE_DONE
