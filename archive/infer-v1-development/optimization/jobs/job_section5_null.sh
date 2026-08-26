#!/usr/bin/env bash
# Archived combined null-validation job.
# Full numerical Section 5 validation; powered 10k null starts only on a pass.

#SBATCH --job-name=sbsi_s5_null
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
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

"$python" "$repo/scripts/run_section5_null.py" \
  --scene-store "$root/scene_store" \
  --measurement-model "$flow" \
  --emulator-metadata "$metadata" --emulator-model "$emulator" \
  --model-cache "$root/model_cache_section5_v1" \
  --proposal-cache "$root/proposal_cache_section5_v1" \
  --output "$root/null_validation_v2_posterior_adapted" \
  --steps 0.005,0.01,0.02 \
  --importance-ladder 8192,32768,65536 \
  --proposal-candidates 131072 --proposal-seeds 8701,8702 \
  --proposal-epsilon 0.1 --proposal-bandwidth 1.0 \
  --n-detected 1024 --oracle-atoms 4096 --oracle-n-detected 256 \
  --bank-column property_seed --max-independent-banks 3 \
  --run-powered-on-pass --powered-n-detected 10000 \
  --object-chunk 16 --atom-chunk 4096 --device cuda
