#!/bin/bash
#SBATCH --job-name=v22nflux
#SBATCH --time=03:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22nflux_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

LOOKUP=results/neighbor_flux_shells_const_c40-139.feather
echo "### V2.2 INTRINSIC NEIGHBOUR-FLUX DIAGNOSTIC job=$SLURM_JOB_ID ###"; date
if [ ! -e "$LOOKUP" ]; then
  "$PY" -u scripts/build_neighbor_flux_shells.py \
    --cases $(seq 40 139) --sign 0.02 --output "$LOOKUP"
fi
"$PY" -u scripts/diag_v22_neighbor_flux.py \
  --dump-glob '/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps/ablate_s2c_lt500_v22_perobj_s*.feather' \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather \
  --flux-lookup "$LOOKUP" --min-case 40 --mag-max 25.8 --re-min 0.5
echo V22_NEIGHBOUR_FLUX_DONE; date
