#!/bin/bash
#SBATCH --job-name=v2own
#SBATCH --time=01:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/v2own_%j.out
set -euo pipefail

# V2 scored on ITS OWN domain (true mag < 26, Re > 0.3) from the SAME 16 dumps that gave
# +1.409% on the V2.2 domain (mag < 25.8, Re > 0.5). Same model, same seeds, same emulator,
# same catalogue -- ONLY the evaluated population changes, so the difference between the two
# numbers is the domain restriction and nothing else.
#
# Expected to reproduce the AGENTS.md fiducial -0.123 +- 0.152%, which is an independent check
# on this whole evaluation path: that number was produced by a different script on a different
# lookup. A mismatch means one of the two paths is wrong.
#
# The certified convention pre-cuts to mag (18,28) and Re (0.1,1.5), so mag<26 & Re>0.3 sits
# entirely inside either candidate emulator box and the coverage check passes.
#
# THE TAG MUST NAME THE EMULATOR WHOSE R_blend IS BAKED INTO THE DUMP, which for
# v2_domain_dumps is `_ho` -- job_s2c_domain_eval.sh builds them with
# blend_lookup_extnbrho_c40-139.feather. Naming `indom_tuned` here would check a box belonging
# to an emulator that produced none of these numbers; it happens to be narrower and so passes
# conservatively, but it is checking the wrong thing. This distinction is exactly why the
# fiducial reads -0.123% in AGENTS.md and -0.271% here: AGENTS.md's number uses the
# indom_tuned lookup (R_blend 0.1358), these dumps carry the `_ho` one (0.1371), and that
# 0.0013 accounts for the 0.150 pt difference in full.

PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
V2=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_domain_dumps

echo "### V2 ON ITS OWN DOMAIN (true mag < 26.0, Re > 0.3) ###"; date
"$PY" -u scripts/eval_v2_indomain_m.py \
  --dump-glob "$V2/ablate_s2c_lt500_dom6x6_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40 --re-min 0.3 --mag-max 26.0 \
  --emulator-tag lsst_r_extnbr_ho --check-emulator-coverage
echo V2_OWN_DOMAIN_DONE; date
