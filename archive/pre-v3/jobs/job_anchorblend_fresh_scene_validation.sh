#!/bin/bash
#SBATCH --job-name=abfr_valid
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abfr_valid_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abfr_valid_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
export PATH=$ENV/bin:$PATH
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/z/Zekang.Zhang/blendemu:$ROOT:${PYTHONPATH:-}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$ROOT"
python -u scripts/fit_anchorblend_coherent_scene_correction.py \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005 \
  --response results/anchorblend_g005_response_v22_c0-99.feather \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext \
  --response results/anchorblend_g005_response_v22_c100-299.feather \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c300-399 \
  --response results/anchorblend_g005_response_v22_c300-399.feather \
  --validation-min 300 --validation-max 399 --fresh-validation \
  --output results/anchorblend_coherent_scene_correction_v22_fresh_c300-399.json
python -u scripts/fit_anchorblend_coherent_scene_correction.py \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005 \
  --response results/anchorblend_g005_response_v22_c0-99.feather \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext \
  --response results/anchorblend_g005_response_v22_c100-299.feather \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c300-399 \
  --response results/anchorblend_g005_response_v22_c300-399.feather \
  --validation-min 300 --validation-max 399 --fresh-validation \
  --feature-set primary_plus \
  --output results/anchorblend_coherent_primaryplus_correction_v22_fresh_c300-399.json
