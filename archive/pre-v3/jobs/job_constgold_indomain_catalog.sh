#!/bin/bash
#SBATCH --job-name=cgind_cat
#SBATCH --time=04:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/cgind_cat_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/cgind_cat_%j.err

set -euo pipefail
BE=/home/z/Zekang.Zhang/blendemu
CFG=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot/configs/fs2_lsst_r_constant_indomain_c40-89.yaml
PY=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
export CONDA_PREFIX=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$BE:${PYTHONPATH:-}"
cd "$BE/scripts"
"$PY" -u run_pipeline.py --config "$CFG" --steps 4 --n-jobs 32 --batch-size 10 \
  --case-offset 40 --n-cases 50
echo CONSTGOLD_INDOMAIN_CATALOG_DONE
