#!/bin/bash
#SBATCH --job-name=mswap_v22_rwvbest
#SBATCH --time=01:30:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/mswap_v22_rwvbest_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/mswap_v22_rwvbest_%j.err
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
LOOKUP=results/blend_lookup_v22_reweighted_vector_optuna30_best_c40-139.feather
OUT=results/constgold_m_swap_v22_reweighted_vector_optuna30_best_16seed.json

export LD_LIBRARY_PATH=/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}
cd "$ROOT"

test -s "$LOOKUP"
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
"$PY" -u scripts/eval_m_swap_emulator_v22.py \
  --dump-glob '/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps/ablate_s2c_lt500_v22_perobj_s*.feather' \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather \
  --lookup "$LOOKUP" \
  --candidate-tag v22_reweighted_vector_optuna30_best_trial15 \
  --flow-checkpoint /project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt \
  --output "$OUT"
test -s "$OUT"
echo V22_REWEIGHTED_VECTOR_BEST_M_SWAP_DONE
