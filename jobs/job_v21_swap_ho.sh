#!/bin/bash
#SBATCH --job-name=v21_swap_ho
#SBATCH --time=02:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/v21_swap_ho_%j.out

# Held-out acceptance gate for a candidate selected without constgold.  On the independent
# all-neighbour half-shear ruler, lsst_r_extnbr_ho is consistent with the summed V2.1 blend truth
# (+1.18% +/- 3.33%), whereas the V2.1-trained emulator is -37.05% +/- 2.07% low.  This fills the
# previously untested cell: V2.1 FLOW dumps with the ruler-preferred _ho R_blend lookup.
#
# FIREWALL: constgold is read here only for this final score.  No parameter is fitted and the
# candidate was selected entirely on the half-shear ruler.
set -euo pipefail
PY=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v21_domain_dumps
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
LOOKUP=results/blend_lookup_extnbrho_c40-139.feather

for f in "$CAT" "$LOOKUP"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done

"$PY" -u scripts/eval_m_swap_emulator.py \
  --dump-glob "$D/ablate_s2c_lt500_v21_perobj_s*.feather" \
  --catalogue "$CAT" --min-case 40 \
  --lookup "$LOOKUP" --lookup-label "lsst_r_extnbr_ho" \
  2>&1 | grep -v --line-buffered "module command"

echo V21_SWAP_HO_DONE
