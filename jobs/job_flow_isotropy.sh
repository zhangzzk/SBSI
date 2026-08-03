#!/bin/bash
#SBATCH --job-name=flow_iso
#SBATCH --time=00:40:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/flow_iso_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/flow_iso_%j.err
#
# Does the trained flow respect the spin-2 rotational symmetry the physics has?  `Pi` on
# rings of constant |e| must be flat if it does.  This localises cont.176's converged
# `<s>_sel` anomaly (138-169 sigma, where §5B.2 predicts zero) to either the model or the
# score machinery; the prior has already been ruled out.
#
#   sbatch $(jobs/pick_gpu.sh) jobs/job_flow_isotropy.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
case "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" in
  *[0-9]Q|*[0-9]A|*[0-9]B) unset PYTORCH_CUDA_ALLOC_CONF ;;
esac

date; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null
python -u scripts/check_flow_isotropy.py \
  --cut-abs-ehat "${CUT:-0.6}" --pi-rows "${PIROWS:-262144}" --reps "${REPS:-4}" \
  --n-phi "${NPHI:-32}" ${EXTRA:-} 2>&1 | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
