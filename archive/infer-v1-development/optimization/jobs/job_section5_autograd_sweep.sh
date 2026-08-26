#!/usr/bin/env bash
# Archived Infer V1 optimization job.
# Exact finite-difference step sweep against Torch autograd.

#SBATCH --job-name=sbsi_s5_autograd
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt
metadata="$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json"
emulator=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/run_section5_autograd_sweep.py" \
  --scene-store "$root/scene_store" \
  --measurement-model "$flow" \
  --emulator-metadata "$metadata" --emulator-model "$emulator" \
  --model-cache "$root/model_cache_section5_v1" \
  --mock "$root/powered_exact_matched_v1/exact_mock" \
  --output "$root/autograd_step_sweep_v1" \
  --exact-atoms 4096 --exact-seed 5201 --objects 32 \
  --steps 0.01,0.005,0.0025,0.00125,0.000625 \
  --object-chunk 32 --atom-chunk 4096 --device cuda
