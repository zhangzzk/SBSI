#!/usr/bin/env bash
# One-case real-flow oracle for catalogue-prior measured selection.
#
# The defaults exercise the established isotropic |e_hat| < 0.6 cut.  Override
# CUT_ABS and CUT_BOUND for magnitude/size selections; an empty CUT_ABS means
# no shape cut.  Every run gets a new output/cache name and refuses overwrite.

#SBATCH --job-name=sbsi_cp_sel
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/realflow_case0_v1/logs/selection_%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/realflow_case0_v1/logs/selection_%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/realflow_case0_v1
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt
metadata="$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json"
emulator=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json

: "${RUN_NAME:=selection_abs06_s8101_n64}"
: "${CUT_ABS=0.6}"
: "${CUT_BOUND:=}"
: "${SELECTION_SAMPLES:=64}"
: "${SELECTION_SEED:=8101}"
: "${PROPOSAL_CACHE:=$root/proposal_cache_alltargets_v2}"

output="$root/$RUN_NAME"
selection_cache="$root/${RUN_NAME}_selection_cache"
if [[ -e "$output" || -e "$selection_cache" ]]; then
  echo "refusing to overwrite $output or $selection_cache" >&2
  exit 2
fi

mkdir -p "$root/logs"
export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cut_args=()
if [[ -n "$CUT_ABS" ]]; then
  cut_args+=(--cut-abs-ehat "$CUT_ABS")
fi
if [[ -n "$CUT_BOUND" ]]; then
  cut_args+=(--cut-bound "$CUT_BOUND")
fi

"$python" "$repo/scripts/run_catalogue_closure.py" \
  --scene-store "$root/scene_store" --measurement-model "$flow" \
  --emulator-metadata "$metadata" --emulator-model "$emulator" \
  --output "$output" --model-cache "$root/model_cache_shapeonly_fd005" \
  --selection-cache "$selection_cache" "${cut_args[@]}" \
  --selection-samples "$SELECTION_SAMPLES" --selection-seed "$SELECTION_SEED" \
  --selection-row-chunk 4096 \
  --proposal-cache "$PROPOSAL_CACHE" --sampler both \
  --n-detected 256 --injected-g1 0.02 --injected-g2 0 \
  --direction-g1 1 --direction-g2 0 --fd-delta 0.005 --no-richardson \
  --proposal-targets measured_ngmix_g1,measured_ngmix_g2,measured_mag_auto,measured_log_flux_radius \
  --proposal-flow-samples 16 --proposal-statistic median \
  --proposal-row-chunk 4096 --proposal-coordinate-seed 7101 \
  --importance-ladder 4096,16384 --proposal-candidates 8192,16384 \
  --proposal-epsilon 0.1 --proposal-bandwidth 1.0 --proposal-seeds 4101,4102 \
  --gate-max-exact-error 0.01 --gate-max-rung-change 0.01 \
  --gate-max-seed-spread 0.01 --gate-max-candidate-change 0.01 \
  --gate-min-ess-fraction 0.02 --gate-max-weight-fraction 0.9 \
  --detection-radius-arcsec 3 --flow-neighbour-radius-arcsec 7 \
  --crowding-near-arcsec 3 --crowding-far-arcsec 7 \
  --pixel-size 0.2 --zero-point 30 --psf-fwhm 0.73 \
  --moffat-beta 2.224 --pixel-rms 0.312 \
  --scene-seed 1401 --detection-seed 2401 --flow-seed 3401 \
  --object-chunk 8 --atom-chunk 2048 --device cuda
