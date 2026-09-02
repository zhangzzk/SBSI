#!/usr/bin/env bash

#SBATCH --job-name=sbsi_flow_train_cmp
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_training_comparison_v1/slurm-%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_training_comparison_v1/slurm-%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_training_comparison_v1

mkdir -p "$root"
export PYTHONPATH="$repo"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

date
nvidia-smi -L
"$python" -u "$repo/scripts/compare_flow_training_distribution.py" \
  --checkpoint /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt \
  --g0-catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather \
  --g005-catalogue /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/det_meas_crowd_conc_g0.05_all200.feather \
  --constgold-sample /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_proxy_topk_size075_n100k_v1/mock_constgold_comparison_100k_v3/constgold_plus_selected_sample_100k.parquet \
  --output "$root/result_v3" \
  --sample-size 100000 \
  --candidate-contexts-per-leg 150000 \
  --feature-bins 3 \
  --permutations 500 \
  --seed 20260901
date
