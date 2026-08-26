#!/usr/bin/env bash
# Archived GalSBI-prior job.
# Build the fixed BlendEMU response for the independent GalSBI scene prior.

#SBATCH --job-name=sbsi_fresh_rb
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_v1/logs/rblend_%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_v1/logs/rblend_%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
: "${PRIOR_ROOT:=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fresh_galsbi_v1}"
root=$PRIOR_ROOT
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt
metadata="$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json"
emulator=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

output="$root/blend_response_v1"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi

"$python" "$repo/scripts/build_catalogue_blend_response.py" \
  --scene-store "$root/scene_store" \
  --measurement-model "$flow" \
  --emulator-metadata "$metadata" --emulator-model "$emulator" \
  --output "$output" \
  --pixel-size 0.2 --zero-point 30 --psf-fwhm 0.73 \
  --moffat-beta 2.224 --pixel-rms 0.312 --device cuda
