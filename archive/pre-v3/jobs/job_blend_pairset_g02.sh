#!/bin/bash
#SBATCH --job-name=blpair02
#SBATCH --time=6:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/blpair02_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/blpair02_%j.err

# STAGE 3/4 of the g=0.2 retrain (WORKLOG 2026-08-03b). Same builder and same g=0 reference leg as
# the g=0.05 pair set, so the two differ in shear amplitude and nothing else. The build's own null
# tests (blend_null, self_null) are the check that the g=0.2 labels are sane -- at 4x the shear the
# nulls should still sit at zero, and if they do not, the retrain must not proceed.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow/blend_pairset_ap7_g02.feather

[ -f "$CAT/det_meas_ngmix_ap7_g0.2_val.feather" ] || { echo "REFUSING: stage 2 output missing"; exit 1; }
echo "### g=0.2 PAIR SET -> $OUT"; date
python -u scripts/build_blend_pairset.py \
    --gs-leg "$CAT/det_meas_ngmix_ap7_g0.2_val.feather" \
    --g0-leg "$CAT/det_meas_ngmix_ap7_g0.0_train.feather" \
    --output "$OUT" || exit 1
echo "BLPAIR02_DONE"; date
