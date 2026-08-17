#!/bin/bash
#SBATCH --job-name=diag5c_probeB3
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_probeB3_%j.out
#
# PROBE B: are the WEIGHTS broken or is the INTEGRAND broken?  Bank, galaxies and posterior
# weights are held exactly as in diag5c_repro.py (job 15459872); only the averaged quantity
# is swapped: phi' / true e1p of the node / an N(0,1) probe independent of everything.  The
# decisive number is whether Var(E_w[noise]) falls like mean(1/ESS_i) as K grows.
#
# Submit:
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --partition=cip --gpus-per-node=a40-16gb:1 \
#       --mem=24G --cpus-per-task=8 jobs/job_diag5c_probeB3.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF

NGAL=${NGAL:-20000}
NODES=${NODES:-6000,20000}
GAMMA=${GAMMA:-0.05}
DELTA=${DELTA:-0.01}
CHUNK=${CHUNK:-24}
SEED=${SEED:-11}
DRAWS=${DRAWS:-20}
BOOT=${BOOT:-300}

date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
stdbuf -oL -eL python -u scripts/diag5c_probeB3_response.py \
  --n-gal "$NGAL" --n-nodes "$NODES" --gamma "$GAMMA" \
  --chunk "$CHUNK" --seed "$SEED" ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "module command"
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
