#!/bin/bash
#SBATCH --job-name=cgind_prep
#SBATCH --time=01:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cgind_prep_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/cgind_prep_%j.err

set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
SOURCE=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
OUTPUT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant_indomain
CASES=({40..89})
export LD_LIBRARY_PATH="/project/ls-gruen/users/zekang.zhang/envs/sims1/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
"$PY" -u scripts/prepare_constgold_indomain.py \
  --original-base "$SOURCE" --output-base "$OUTPUT" --cases "${CASES[@]}" \
  --mag-min 18 --mag-max 28 --re-min 0.1 --re-max 1.5
echo CONSTGOLD_INDOMAIN_PREP_DONE
