#!/bin/bash
#SBATCH --job-name=hsg05_vec
#SBATCH --time=01:30:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hsg05_vec_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hsg05_vec_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
TABLE=/project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_vector_g005_c0-39.feather
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:$ROOT/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
python -u scripts/build_halfshear_vector_closure.py \
  --catalogue /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_g005_train.feather \
  --shear 0.05 --n-cases 40 --table-output "$TABLE" \
  --output results/halfshear_vector_closure_v22_g005_c0-39.json
echo HALFSHEAR_G005_VECTOR_CLOSURE_JOB_DONE
date
