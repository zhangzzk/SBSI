#!/bin/bash
#SBATCH --job-name=cgind_ana
#SBATCH --time=01:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cgind_ana_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/cgind_ana_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
NEW=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_constgold_indomain_c40-89/ablate_s2c_lt500_v22_indomain_perobj_s{seed}.feather
OLD=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps/ablate_s2c_lt500_v22_perobj_s{seed}.feather
MANIFEST=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant_indomain/indomain_filter_manifest.json
NEWCAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant_indomain/constant_response_catalogue_train.feather
OLDCAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
OUT=$ROOT/results/v22_constgold_indomain_c40-89.json
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
[ ! -e "$OUT" ] || { echo "REFUSING overwrite $OUT"; exit 1; }
"$PY" -u scripts/analyze_constgold_indomain.py \
  --new-pattern "$NEW" --old-pattern "$OLD" --new-catalogue "$NEWCAT" --old-catalogue "$OLDCAT" \
  --min-case 40 --max-case 90 --primary-mag-max 25.8 --primary-re-min 0.5 \
  --n-boot 2000 --manifest "$MANIFEST" --output "$OUT"
echo CONSTGOLD_INDOMAIN_ANALYZE_DONE
