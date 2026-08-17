#!/bin/bash
#SBATCH --job-name=v22cg_loc
#SBATCH --time=01:30:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22cg_loc_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22cg_loc_%j.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
cd "$ROOT"
"$PY" -u scripts/localize_v22_constgold_gap.py \
  --dump-glob '/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps/ablate_s2c_lt500_v22_perobj_s*.feather' \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather \
  --dominance-lookup results/v22_constgold_response_dominance_c40-139.feather \
  --output-feather results/v22_constgold_gap_features_c40-139.feather \
  --output-json results/v22_constgold_gap_localization_c40-139.json \
  --output-md results/v22_constgold_gap_localization_c40-139.md
