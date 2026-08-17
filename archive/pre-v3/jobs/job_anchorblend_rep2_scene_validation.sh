#!/bin/bash
#SBATCH --job-name=abrep2_scene
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abrep2_scene_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abrep2_scene_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
REP=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchorblend_g005_response_v22_c400-599.feather
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
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c400-599 \
  --response "$REP" \
  --validation-min 400 --validation-max 599 --fresh-validation \
  --output results/anchorblend_coherent_scene_correction_v22_rep2_c400-599.json
python -u scripts/fit_anchorblend_coherent_scene_correction.py \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005 \
  --response results/anchorblend_g005_response_v22_c0-99.feather \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_ext \
  --response results/anchorblend_g005_response_v22_c100-299.feather \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c400-599 \
  --response "$REP" \
  --validation-min 400 --validation-max 599 --fresh-validation \
  --feature-set primary_plus \
  --output results/anchorblend_coherent_primaryplus_correction_v22_rep2_c400-599.json
echo ANCHORBLEND_REP2_SCENE_VALIDATION_DONE
