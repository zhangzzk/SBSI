#!/usr/bin/env bash
# Archived Infer V1 optimization job.
# Per-object finite-shear curvature and posterior-mixture decomposition.

#SBATCH --job-name=sbsi_s5_curvature
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1

: "${FLOW:=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt}"
: "${EMULATOR_METADATA:=$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json}"
: "${EMULATOR_MODEL:=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json}"
: "${MODEL_CACHE:=$root/model_cache_section5_v1}"
: "${PROPOSAL_CACHE:=$root/proposal_cache_section5_v1}"

: "${MOCK_INPUT:?set MOCK_INPUT to a saved mock directory}"
: "${OUTPUT:?set OUTPUT to a new output directory}"
: "${DIRECTION:?set DIRECTION to g1 or g2}"
: "${CENTERS:=0,0.01,0.02}"
: "${PREVIOUS_DRAWS:=8192}"
: "${DRAWS:=16384}"
: "${PROPOSAL_CANDIDATES:=32768}"
: "${PROPOSAL_SEED:=8701}"
: "${H:=0.00125}"
: "${ZERO_DERIVATIVE_STEPS:=0.00125,0.0025}"
: "${OBJECT_CHUNK:=4}"
: "${DECOMPOSITION_CHUNK:=32}"

case "$DIRECTION" in
  g1|g2) ;;
  *) echo "DIRECTION must be g1 or g2" >&2; exit 2 ;;
esac

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/run_section5_curvature_scan.py" \
  --scene-store "$root/scene_store" \
  --measurement-model "$FLOW" \
  --emulator-metadata "$EMULATOR_METADATA" --emulator-model "$EMULATOR_MODEL" \
  --model-cache "$MODEL_CACHE" \
  --proposal-cache "$PROPOSAL_CACHE" \
  --mock-input "$MOCK_INPUT" --output "$OUTPUT" \
  --direction "$DIRECTION" --centers "$CENTERS" \
  --h "$H" --zero-derivative-steps "$ZERO_DERIVATIVE_STEPS" \
  --previous-draws "$PREVIOUS_DRAWS" --draws "$DRAWS" \
  --proposal-candidates "$PROPOSAL_CANDIDATES" \
  --proposal-seed "$PROPOSAL_SEED" --proposal-epsilon 0.1 \
  --object-chunk "$OBJECT_CHUNK" --decomposition-chunk "$DECOMPOSITION_CHUNK" \
  --device cuda
