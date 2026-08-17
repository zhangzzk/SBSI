#!/bin/bash
#SBATCH --job-name=v30_fullrw
#SBATCH --time=03:00:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v30_fullrw_%j.out
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$ROOT"

C=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v30_nosize_grid
V29=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v29_conditional
F=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v27_scene/v28_rblend
OUT=results/v30_full_conditioner_population_reweight.json
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }

"$PY" -u scripts/diag_full_conditioner_reweight.py \
  --features-half "$F/halfshear_rblend_base_c40-199_v22domain.feather" \
  --features-const "$V29/const_primary_features_c40-139.feather" \
  --const-catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather \
  --crowd-lookup /home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather \
  --constgold-glob "$C/constgold_scores/v30_m10_c16_s501_c*.feather" \
  --baseline-template "$V29/halfshear_scores/v22_control_s{seed}.feather" \
  --variant-dir "$C/halfshear_scores" \
  --arms v30_m10_c16 v30_m12_c16 --seeds 501 502 503 505 \
  --closure-json results/v30_nosize_grid_fourseed_screen.json \
  --min-case 40 --max-case 140 --folds 5 --train-per-domain 350000 \
  --output "$OUT"

echo V30_FULLRW_DONE; date
