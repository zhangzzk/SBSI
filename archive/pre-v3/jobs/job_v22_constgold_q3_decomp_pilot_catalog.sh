#!/bin/bash
#SBATCH --job-name=v22q3_cat
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22q3_cat_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22q3_cat_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
BE=/home/z/Zekang.Zhang/blendemu
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ORIGINAL=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
LOOKUP=$ROOT/results/blend_lookup_v22_c40-139.feather
PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_v22cgq3_pilot
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_constgold_q3_decomp_pilot
BASE_CONFIG=$BE/configs/base_sim_config.ini
NOISE=/project/ls-gruen/users/zekang.zhang/lsst_sims/Euclid_Q1_median_LSST10yr.csv
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:$BE:${PYTHONPATH:-}"
export XGB_DEVICE=cpu
cd "$ROOT"
"$PY" -u scripts/prepare_v22_constgold_bin_decomposition.py \
  --base "total=${PREFIX}_total" \
  --base "self=${PREFIX}_self" \
  --base "deployed=${PREFIX}_deployed" \
  --base "other=${PREFIX}_other" \
  --original-base "$ORIGINAL" --lookup "$LOOKUP" --manifest-dir "$MANIFEST" \
  --cases 40 41 42 43 44 45 46 47 --g 0.02 \
  --scene-radius 15.0 --min-separation 30.01 --tag lsst_r_extnbr_v22 \
  --base-config "$BASE_CONFIG" --noise-csv "$NOISE" --noise-band LSST_r \
  --mag-cut 29 --pixel-scale 0.2
echo V22_CONSTGOLD_Q3_DECOMP_PILOT_CATALOG_DONE
