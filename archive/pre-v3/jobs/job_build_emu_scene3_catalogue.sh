#!/bin/bash
#SBATCH --job-name=emu_s3cat
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/emu_s3cat_%j.out
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu:/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:${PYTHONPATH:-}"
SRC=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather
LOW=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/results/neighbor_flux_shells_hs_c0-39.feather
HIGH=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/results/neighbor_flux_shells_hs_c40-199.feather
OUTDIR=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_scene3; OUT=$OUTDIR/response_catalogue_train.feather
mkdir -p "$OUTDIR"; for f in "$SRC" "$LOW" "$HIGH"; do [ -f "$f" ] || { echo "MISSING $f"; exit 1; }; done
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
cd /home/z/Zekang.Zhang/blendemu
echo "### BUILD V2.2 SCENE3 RESPONSE CATALOGUE job=$SLURM_JOB_ID ###"; date
"$PY" -u scripts/augment_response_scene_flux.py --response "$SRC" --lookups "$LOW" "$HIGH" --output "$OUT" --case-min 0
echo EMU_SCENE3_CATALOGUE_DONE; date
