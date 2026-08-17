#!/bin/bash
#SBATCH --job-name=hs_vec_train
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/hs_vec_train_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/hs_vec_train_%j.err
set -euo pipefail

ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876/response_catalogue_train.feather
TABLE=/project/ls-gruen/users/zekang.zhang/sbsi_gap_halfshear_vector_closure_v22_training_c40-199.feather
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:$ROOT/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
python -u scripts/build_halfshear_vector_closure.py \
  --catalogue "$CAT" --case-min 40 --n-cases 200 --development-max 119 \
  --shear 0.2 --model-tag lsst_r_extnbr_v22 \
  --table-output "$TABLE" \
  --output results/halfshear_vector_closure_v22_training_c40-199.json
echo HALFSHEAR_VECTOR_CLOSURE_TRAINING_DONE
