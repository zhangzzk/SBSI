#!/usr/bin/env bash
#SBATCH --job-name=sbsi_uparentprep
#SBATCH --partition=inter,cip
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v1
raw=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876
crowd=/project/ls-gruen/users/zekang.zhang/sbsi_caches/crowd_flux_conc_c0-199.feather

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
"$python" -u -m scripts.prepare_full_parent_classifier \
  --raw-simulation-root "$raw" \
  --crowd-catalogue "$crowd" \
  --output-root "$root/classifier_prepared" \
  --case-start 40 --case-stop 120
