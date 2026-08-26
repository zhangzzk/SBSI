#!/usr/bin/env bash
# Generate paired Section 5 mocks without running the downstream likelihood closure.

#SBATCH --job-name=sbsi_s5_mocks
#SBATCH --partition=inter,cip
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/section5_galsbi_100cases_v1

: "${OUTPUT:?set OUTPUT to a new output directory}"
: "${FLOW:=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt}"
: "${EMULATOR_METADATA:=$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json}"
: "${EMULATOR_MODEL:=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json}"
: "${MODEL_CACHE:=$root/model_cache_section5_v1}"
: "${PROPOSAL_CACHE:=$root/proposal_cache_section5_v1}"
: "${N_DETECTED:=10000}"
: "${INJECTION:=0.02}"
: "${COMPONENTS:=g1 g2}"

case "$COMPONENTS" in
  "g1"|"g2"|"g1 g2"|"g2 g1") ;;
  *) echo "COMPONENTS must be 'g1', 'g2', or 'g1 g2'" >&2; exit 2 ;;
esac

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# COMPONENTS is validated above and deliberately split into CLI words.
# shellcheck disable=SC2086
"$python" "$repo/scripts/run_section5_nonzero_closure.py" \
  --scene-store "$root/scene_store" \
  --measurement-model "$FLOW" \
  --emulator-metadata "$EMULATOR_METADATA" --emulator-model "$EMULATOR_MODEL" \
  --model-cache "$MODEL_CACHE" --proposal-cache "$PROPOSAL_CACHE" \
  --output "$OUTPUT" --n-detected "$N_DETECTED" \
  --injection "$INJECTION" --components $COMPONENTS \
  --scene-seed 2101 --detection-seed 3101 --flow-seed 4101 \
  --orientation-seed 5101 --generate-only --device cuda
