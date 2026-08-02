#!/bin/bash
#SBATCH --job-name=score_select
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/score_select_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/score_select_%j.err
#
# §5B end to end WITH selection: the full (5.3) on the real flow against a known shear.
# Closes cont.164 defect 2 (no I_sel) on real data; A.7 prices its omission at m = -64%.
# The cut is |xhat| < c -- isotropic, so §5B.2 predicts <s>_sel ~ 0 and I_sel carries all.
#
# Submit:
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       jobs/job_score_select.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF

G=${G:-0.02}
CUT=${CUT:-0.6}
ROWS=${ROWS:-400000}
PIROWS=${PIROWS:-4096}          # comma list sweeps the population sample off ONE score pass
PISAMP=${PISAMP:-8}
RING=${RING:-rot90}             # 90-degree ring pairs: the shape-noise variance reduction

date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/eval_score_select.py \
  --closure-g "$G" --cut-abs-ehat "$CUT" --max-rows "$ROWS" \
  --pi-rows "$PIROWS" --pi-samples "$PISAMP" --pi-reps "${PIREPS:-4}" \
  --ring "$RING" --shape-reps "${SHAPEREPS:-1}" \
  --jk-blocks "${JKBLOCKS:-200}" --uncut-control ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
