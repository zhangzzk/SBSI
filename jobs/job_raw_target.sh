#!/bin/bash
#SBATCH --job-name=raw_target
#SBATCH --time=01:30:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/raw_target_%j.out

# UNIFY-THE-ESTIMATOR (cont.115 owner call): rebuild the flow response target as the RAW
# <e_ngmix.ghat_p>/g response (NO SNC g=0 subtraction) on the SAME det_meas_ngmix_g0.05 leg the
# ngmix->constgold BRIDGE was measured on. Binned (flux x size x blend) -> per-cell mean is
# low-variance despite raw per-object noise. This puts the flow target, the bridge reference, and
# (via the bridge) the constgold antithetic truth on ONE footing, removing the SNC-vs-raw ~10% gap
# that made m_iso=+13.5%. Same binning as response_target_isoblend_snc (6 x [3 size] x [iso+4 dist]).
# FIREWALL: half-shear ngmix leg only; constgold never read.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI

CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.05_val.feather
OUT=results/response_target_isoblend_RAW_c0-99_6x3x5.npz
echo "### RAW_TARGET job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -B -u scripts/compute_response_target_blend.py \
  --catalogue "$CAT" \
  --nominal-g 0.05 \
  --n-flux 6 --size-edges 0.1,0.2407,0.4117,1.5 --n-dist 4 \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --min-count 200 --max-rows 40000000 \
  --output "$OUT" \
  || { echo "RAW_TARGET FAILED"; exit 1; }
echo "### RAW_TARGET_DONE job=$SLURM_JOB_ID ###"; date