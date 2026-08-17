#!/bin/bash
#SBATCH --job-name=finesz_tgt
#SBATCH --time=01:30:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/finesz_tgt_%j.out

# ROOT-CAUSE FIX (cont.116): the flow's R_self(size) is mis-shaped because the response target's
# SIZE binning is too COARSE — the steep transition [0.24,0.41] is ONE bin (+0.21) while the true
# response rises 0->0.95 across it, so the flow only learns the coarse average. Rebuild with FINE
# size bins through the transition (0.24-0.8) so the response loss resolves the true size-slope.
# RAW estimator on the ngmix g0.05 leg (matches the bridge reference). FIREWALL: half-shear only.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.05_val.feather
OUT=results/response_target_isoblend_RAWfine_c0-99_6x9x5.npz
echo "### FINESZ_TGT job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -B -u scripts/compute_response_target_blend.py \
  --catalogue "$CAT" \
  --nominal-g 0.05 \
  --n-flux 6 --size-edges 0.1,0.24,0.30,0.36,0.42,0.50,0.62,0.80,1.10,1.50 --n-dist 4 \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --min-count 200 --max-rows 40000000 \
  --output "$OUT" \
  || { echo "FINESZ_TGT FAILED"; exit 1; }
echo "### FINESZ_TGT_DONE job=$SLURM_JOB_ID ###"; date