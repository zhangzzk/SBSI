#!/bin/bash
#SBATCH --job-name=hsnflux
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsnflux_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

OUT=results/neighbor_flux_shells_hs_c40-199.feather
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
echo "### HALF-SHEAR INTRINSIC NEIGHBOUR FLUX cases 40-199 job=$SLURM_JOB_ID ###"; date
"$PY" -u scripts/build_neighbor_flux_shells.py \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --sign 0.0 --cases $(seq 40 199) --output "$OUT"
echo HS_NEIGHBOR_FLUX_DONE; date
