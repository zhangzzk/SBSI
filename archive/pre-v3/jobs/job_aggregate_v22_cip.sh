#!/bin/bash
#SBATCH --job-name=agg22_cip
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/agg22_cip_%j.out
set -euo pipefail

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
V22=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps
V2=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps

echo "### V2.2 CIP-SHARDED SEED-ENSEMBLE RESULT ###"; date
"$PY" -u scripts/eval_v2_indomain_m.py \
  --dump-glob "$V22/ablate_s2c_lt500_v22_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40 --re-min 0.5 --mag-max 25.8 \
  --emulator-tag lsst_r_extnbr_v22 --check-emulator-coverage

# The tag must name the emulator whose R_blend is IN the dump: v2_domain_dumps are built with
# blend_lookup_extnbrho (job_s2c_domain_eval.sh), so it is `_ho`, not indom_tuned. The earlier
# indom_tuned label checked a box belonging to an emulator that produced none of these numbers.
# It passed only because that box is narrower; the reported m was never affected, since the tag
# drives the coverage check alone and R_blend is read from the dump.
echo "### V2 MODEL ON THE SAME V2.2 POPULATION (population-only control) ###"
"$PY" -u scripts/eval_v2_indomain_m.py \
  --dump-glob "$V2/ablate_s2c_lt500_dom6x6_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40 --re-min 0.5 --mag-max 25.8 \
  --emulator-tag lsst_r_extnbr_ho --check-emulator-coverage
echo AGG_V22_CIP_DONE; date
