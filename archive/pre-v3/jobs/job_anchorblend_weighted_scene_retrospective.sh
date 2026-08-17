#!/bin/bash
#SBATCH --job-name=ab_wscene_old
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ab_wscene_old_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/ab_wscene_old_%j.err
set -euo pipefail
ROOT=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
ENV=/project/ls-gruen/users/zekang.zhang/envs/sims1
TAG=lsst_r_extnbr_v22_rpowa0065
PARTS=/project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpowa0065_combined_c0-399
mkdir -p "$PARTS/$TAG"
for case in $(seq 0 199); do
  LINK="$PARTS/$TAG/case${case}.feather"
  test -L "$LINK" || ln -s /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpowa0065_c0-199/$TAG/case${case}.feather "$LINK"
done
for case in $(seq 200 299); do
  LINK="$PARTS/$TAG/case${case}.feather"
  test -L "$LINK" || ln -s /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpow_final_c200-299/$TAG/case${case}.feather "$LINK"
done
for case in $(seq 300 399); do
  LINK="$PARTS/$TAG/case${case}.feather"
  test -L "$LINK" || ln -s /project/ls-gruen/users/zekang.zhang/sbsi_gap_anchor_rpow_final_c300-399/$TAG/case${case}.feather "$LINK"
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
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_anchorblend_g005_c300-399 \
  --response results/anchorblend_g005_response_v22_c300-399.feather \
  --validation-min 200 --validation-max 399 \
  --candidate-parts-root "$PARTS" --candidate-tag "$TAG" \
  --output results/anchorblend_weighted_scene_correction_v22_retrospective_c200-399.json
echo ANCHORBLEND_WEIGHTED_SCENE_RETROSPECTIVE_DONE
