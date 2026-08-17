#!/bin/bash
#SBATCH --job-name=mswap_v2_rwvfix
#SBATCH --time=01:30:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/mswap_v2_rwvfix_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/mswap_v2_rwvfix_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
LOOKUP=results/blend_lookup_v2_reweighted_vector_fixed_c40-139.feather
OUT=results/constgold_m_swap_v2_reweighted_vector_fixed_16seed.json

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"

test -s "$LOOKUP"
[ ! -e "$OUT" ] || { echo "REFUSING to overwrite $OUT"; exit 1; }
"$PY" -u scripts/eval_m_swap_emulator_v22.py \
  --dump-glob '/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps/ablate_s2c_lt500_dom6x6_perobj_s*.feather' \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather \
  --lookup "$LOOKUP" \
  --candidate-tag v2_reweighted_vector_fixed_from_v22_trial15 \
  --flow-checkpoint /project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s501_swaavg.pt \
  --primary-mag-max 26.0 \
  --primary-re-min 0.3 \
  --domain-label 'V2 rectangular primary domain' \
  --output "$OUT"
test -s "$OUT"
echo V2_REWEIGHTED_VECTOR_FIXED_M_SWAP_DONE
