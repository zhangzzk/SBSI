#!/bin/bash
#SBATCH --job-name=pi_grid
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/pi_grid_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/pi_grid_%j.err
#
# Refine the POPULATION block's node bank while leaving the score pass (and therefore the
# 5.7 h score cache) untouched.  scripts/check_quadrature.py measured the Pi-weighted
# integral to be 2.4% off at the production grid_n=61 -- 0.90% on `m`, larger than the whole
# residual -- while the per-galaxy machinery is already good to 0.04%.  Since the two are
# different integrals, only the population one needs refining, and since the Pi fast path it
# costs minutes.  If d(m) walks toward zero along this ladder, closure is a quadrature
# artefact and this is the fix.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
case "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" in
  *[0-9]Q|*[0-9]A|*[0-9]B) unset PYTORCH_CUDA_ALLOC_CONF ;;
esac
SC=/project/ls-gruen/users/zekang.zhang/sbsi_scores
date; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
for CUT in ${CUTS:-0.6 0.4}; do
  TAG=$(echo "$CUT" | tr -d '.' | sed 's/^0//')
  for PG in ${PGRIDS:-61 81 101 141}; do
    echo ""; echo "######## cut=$CUT  pi_grid_n=$PG ########"
    python -u scripts/eval_score_select.py \
      --closure-g 0.05 --cut-abs-ehat "$CUT" --max-rows 2000000 \
      --pi-rows 1048576 --pi-samples 8 --pi-reps 6 \
      --ring rot90 --shape-reps 2 --jk-blocks 200 --uncut-control \
      --pi-grid-n "$PG" \
      --load-scores "$SC/c0${TAG}_g05_8M_G2765.npz" 2>&1 \
      | grep -vE "module command|Pi rep "
  done
done
date; echo "### DONE ###"
