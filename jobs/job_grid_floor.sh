#!/bin/bash
#SBATCH --job-name=grid_floor
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/grid_floor_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/grid_floor_%j.err
#
# Predict how much of the uncut-control bias the SCORE grid alone can explain, by running the
# closure test with the flow replaced by a Gaussian shape likelihood -- flow-free, data-free,
# CPU only.  See scripts/predict_grid_floor.py for what this can and cannot claim.  The whole
# point is that it makes a NUMBER for the paired job (grid_n 61 -> 101) to be tested against,
# rather than the "right sign, enough room" upper bound cont.182 had.
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate py31
REPO=${REPO:-/home/z/Zekang.Zhang/SBSI}
cd "$REPO" || exit 1
export PYTHONPATH="$REPO:$PYTHONPATH"
# numpy's einsum threads through BLAS only for the matmul; give it the cores anyway.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
PRIOR=${PRIOR:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/etilde_prior_e_samples_dom25805.feather}
date
python -u scripts/predict_grid_floor.py \
  --prior-sample "$PRIOR" \
  --objects "${OBJECTS:-50000}" --jk-blocks "${JKB:-100}" \
  --grid-ns "${GRIDNS:-61:81:101:141}" --reference-n "${REFN:-201}" \
  --sigma-sweep "${SIGMAS:-0.20:0.25:0.30}" \
  --closure-g "${CLOSURE_G:-0.05}" --chunk "${CHUNK:-256}"
date; echo "### DONE ###"
