#!/bin/bash
#SBATCH --job-name=v22_sameobj
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22_sameobj_%j.out
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
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps
OUT=results/v22_same_object_closure_c40-139_s16.json
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }

"$PY" -u scripts/diag_v22_same_object_closure.py \
  --const-domain-glob "$C/constgold_scores/v30_m10_c16_s501_c*.feather" \
  --const-primary-features "$V29/const_primary_features_c40-139.feather" \
  --const-crowd-lookup /home/z/Zekang.Zhang/SBSI/results/crowd_flux_conc_c0-199.feather \
  --const-dump-template "$D/ablate_s2c_lt500_v22_perobj_s{seed}.feather" \
  --half-features "$F/halfshear_rblend_base_c40-199_v22domain.feather" \
  --half-selfresp /project/ls-gruen/users/zekang.zhang/sbsi_caches/hs_selfresp_v22_c40-199/halfshear_selfresp_v22_c40-199.feather \
  --seeds 501 502 503 505 506 507 508 509 510 511 512 513 514 515 516 517 \
  --min-case 40 --max-case 140 \
  --output "$OUT"

echo V22_SAME_OBJECT_DONE; date
