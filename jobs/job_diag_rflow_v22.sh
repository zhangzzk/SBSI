#!/bin/bash
#SBATCH --job-name=rfv22
#SBATCH --time=01:30:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rfv22_%j.out
set -euo pipefail

# WHERE in the V2.2 domain does the +1.081% live? The 16-seed number is a single average over
# 5,642,350 rows; this bins the closure residual by true size, S/N and magnitude so a flat
# residual can be told from one concentrated in a corner of the population.
#
# Scope is the V2.2 box (mag < 25.8, Re > 0.5) via --scope train with those bounds, so the tables
# span the box rather than the V2.1 curve.
#
# R_blend is left as the dump's own (the v22 emulator). That is deliberate: the question here is
# where the ACTUAL V2.2 model's residual sits, not a flow-vs-flow comparison, so the model must be
# scored as deployed. The independent judgement of that emulator is the per-pair ruler
# (jobs 15595420 / 15595421), never this table -- constgold cannot arbitrate R_blend.
#
# CAVEAT ON READING THE OUTPUT: the script's final "cell" table is hardcoded to the V2.1 cut
# (Re > 0.5 AND sn > 10). V2.2's second condition is mag < 25.8, NOT S/N, so that last table does
# not describe the V2.2 domain and must be ignored. The size / S/N / magnitude tables above it are
# generic and valid.

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
V22=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps

echo "### V2.2 CLOSURE RESIDUAL, BINNED (16 seeds) ###"; date
"$PY" -u scripts/diag_rflow_v21.py --scope train \
  --dump-glob "$V22/ablate_s2c_lt500_v22_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40 \
  --mag-max 25.8 --re-min 0.5
echo RFV22_DONE; date
