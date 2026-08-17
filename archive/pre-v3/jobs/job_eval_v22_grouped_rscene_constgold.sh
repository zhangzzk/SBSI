#!/bin/bash
#SBATCH --job-name=v22grs_cg_eval
#SBATCH --time=01:30:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22grs_cg_eval_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22grs_cg_eval_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
RUN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_grouped_rscene_v1
LOOKUP=$RUN/constgold_c40-139/lookup.feather
OUT=$ROOT/results/constgold_m_swap_v22_grouped_rscene_16seed.json
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
for input in "$LOOKUP"; do
  test -s "$input" || { echo "MISSING $input"; exit 1; }
done
test ! -e "$OUT" || { echo "REFUSING existing $OUT"; exit 1; }
cd "$ROOT"
python -u scripts/eval_m_swap_emulator_v22.py \
  --dump-glob '/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps/ablate_s2c_lt500_v22_perobj_s*.feather' \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather \
  --lookup "$LOOKUP" \
  --candidate-tag v22_grouped_rscene_final_c40-199 \
  --flow-checkpoint /project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt \
  --output "$OUT"
test -s "$OUT"
echo V22_GROUPED_RSCENE_CONSTGOLD_EVAL_DONE
date
