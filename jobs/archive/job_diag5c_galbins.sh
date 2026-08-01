#!/bin/bash
#SBATCH --job-name=diag5c_galbins
#SBATCH --time=01:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/diag5c_galbins_%j.out
# GALAXY-LEVEL anatomy of the 5C information-equality failure + a SECOND checkpoint.
#   NO_EXPANDABLE_SEGMENTS=1 sbatch --gpus-per-node=a40-16gb:1 --mem=24G \
#       --cpus-per-task=8 jobs/job_diag5c_galbins.sh
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI/.claude/worktrees/inference-5b}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
[ "${NO_EXPANDABLE_SEGMENTS:-0}" = "1" ] && unset PYTORCH_CUDA_ALLOC_CONF
CKDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
CKS=${CKS:-$CKDIR/forward_ens_lr250_swa8_seed421_joint.pt,$CKDIR/forward_ens_lr250_swa8_seed422_joint.pt,$CKDIR/forward_ensH_nc_swa8_seed421_joint.pt}
NGAL=${NGAL:-4000}; NNODE=${NNODE:-2000}; OFF=${OFF:-20000}
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
for GG in ${GAMMAS:-0.0 0.05}; do
  echo; echo "##################### gamma=$GG #####################"
  stdbuf -oL -eL python -u scripts/diag5c_galbins.py \
    --checkpoints "$CKS" --n-gal "$NGAL" --n-node "$NNODE" --gal-offset "$OFF" \
    --gamma "$GG" ${EXTRA:-} 2>&1 | grep --line-buffered -vE "module command"
done
STATUS=$?; date; echo "### DONE (exit $STATUS) ###"; exit $STATUS
