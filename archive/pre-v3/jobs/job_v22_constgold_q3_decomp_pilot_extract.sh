#!/bin/bash
#SBATCH --job-name=v22q3_ext
#SBATCH --time=04:00:00
#SBATCH --mem=240G
#SBATCH --cpus-per-task=12
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v22q3_ext_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/v22q3_ext_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
PREFIX=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_v22cgq3_pilot
MANIFEST=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_constgold_q3_decomp_pilot
DUMP=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_cip_domain_dumps/ablate_s2c_lt500_v22_perobj_s501.feather
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_constgold_q3_decomp_pilot_truth.feather
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:/home/z/Zekang.Zhang/blendemu:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -u scripts/extract_v22_constgold_bin_decomposition.py \
  --base "total=${PREFIX}_total" \
  --base "self=${PREFIX}_self" \
  --base "deployed=${PREFIX}_deployed" \
  --base "other=${PREFIX}_other" \
  --manifest-dir "$MANIFEST" --evaluation-dump "$DUMP" \
  --cases 40 41 42 43 44 45 46 47 \
  --g 0.02 --min-coverage 0.25 --output "$OUT"
echo V22_CONSTGOLD_Q3_DECOMP_PILOT_EXTRACT_DONE
