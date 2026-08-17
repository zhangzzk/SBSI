#!/bin/bash
#SBATCH --job-name=abrep2_wscene
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/abrep2_wscene_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/abrep2_wscene_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
TAG=lsst_r_extnbr_v22_rpowa0065
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpowa0065_combined_c0-199_c400-599
mkdir -p "$PARTS/$TAG"
for case in $(seq 0 199); do
  ln -s /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpowa0065_c0-199/$TAG/case${case}.feather "$PARTS/$TAG/case${case}.feather"
done
for case in $(seq 400 599); do
  ln -s /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpowa0065_c400-599/$TAG/case${case}.feather "$PARTS/$TAG/case${case}.feather"
done
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
  --response /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchorblend_g005_response_v22_c400-599.feather \
  --validation-min 400 --validation-max 599 --fresh-validation \
  --candidate-parts-root "$PARTS" --candidate-tag "$TAG" \
  --output results/anchorblend_weighted_scene_correction_v22_rep2_c400-599.json
echo ANCHORBLEND_REP2_WEIGHTED_SCENE_VALIDATION_DONE
