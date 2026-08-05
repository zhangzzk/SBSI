#!/bin/bash
#SBATCH --job-name=v21bfm
#SBATCH --time=03:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/v21bfm_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v21bfm_%j.err

# Parameter-free acceptance test of the strongest existing firewall-clean model change:
# replace only V2.1's BlendEMU term with the independently trained 16-seed g=0.2 flow-#2 term.
# All model choices predate this test and were made on half-shear rulers. Constgold is read only
# here, for final evaluation; no output is fitted, scaled, or fed back into either model.
set -euo pipefail
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python

CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
DUMPS=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v21_domain_dumps/ablate_s2c_lt500_v21_perobj_s\*.feather
LOOKUP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/blendflow/blend_lookup_ens16g02_allpairs7_c40-139.feather

"$PY" -u scripts/eval_v21_blendflow.py \
  --dump-glob "$DUMPS" --catalogue "$CAT" --flow2-lookup "$LOOKUP"

echo V21BFM_DONE
