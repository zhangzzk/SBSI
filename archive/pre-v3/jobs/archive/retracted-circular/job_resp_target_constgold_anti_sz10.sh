#!/bin/bash
#SBATCH --job-name=rtgt_cg10
#SBATCH --time=01:30:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=10
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/rtgt_cg10_%j.out
# FINER-size (n_size=10) metric-consistent constgold antithetic target: addresses the now-dominant
# WITHIN-CELL size gradient (size0.5-1.0 +1.8% vs size1.0-1.5 -2.3% straddling the top bin) left after
# the sz6cg scheme fix. 6x10 flux x size, isolated + 4 distance bins. Cases 0-99. FIREWALL: a-priori.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
CG=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
OUT=results/response_target_constgold_c0-99_6x10x5.npz
echo "### RTGT_CG10 job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/compute_response_target_blend.py \
  --catalogue "$CG" --antithetic --nominal-g 0.02 \
  --n-flux 6 --n-size 10 --n-dist 4 --max-case 99 --min-count 200 --output "$OUT" \
  || { echo "RTGT_CG10 FAILED"; exit 1; }
echo "### RTGT_CG10_DONE job=$SLURM_JOB_ID ###"; ls -la "$OUT"; date
