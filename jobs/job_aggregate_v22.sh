#!/bin/bash
#SBATCH --job-name=agg_v22
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/agg_v22_%j.out
set -euo pipefail

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
V22=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_domain_dumps
V2=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps

echo "### V2.2 TWO-SEED RESULT ###"; date
python -u scripts/eval_v2_indomain_m.py \
  --dump-glob "$V22/ablate_s2c_lt500_v22_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40 --re-min 0.5 --mag-max 25.8 \
  --emulator-tag lsst_r_extnbr_v22 --check-emulator-coverage

echo "### V2 MODEL ON THE SAME V2.2 POPULATION (population-only control) ###"
python -u scripts/eval_v2_indomain_m.py \
  --dump-glob "$V2/ablate_s2c_lt500_dom6x6_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40 --re-min 0.5 --mag-max 25.8 \
  --emulator-tag lsst_r_extnbr_indom_tuned --check-emulator-coverage
echo AGG_V22_DONE; date
