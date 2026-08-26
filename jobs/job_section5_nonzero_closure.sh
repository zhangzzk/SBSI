#!/usr/bin/env bash
# Paired local nonzero-shear closure with common mock random streams.

#SBATCH --job-name=sbsi_s5_nonzero
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
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt
metadata="$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json"
emulator=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json

: "${OUTPUT:=$root/nonzero_paired_n10000_g000125_h00125_v1}"
: "${N_DETECTED:=10000}"
: "${INJECTION:=0.00125}"
: "${COMPONENTS:=g1 g2}"
: "${PREVIOUS_DRAWS:=32768}"
: "${DRAWS:=65536}"
: "${PROPOSAL_CANDIDATES:=131072}"
: "${PROPOSAL_SEED:=8701}"

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
  --measurement-model "$flow" \
  --emulator-metadata "$metadata" --emulator-model "$emulator" \
  --model-cache "$root/model_cache_section5_v1" \
  --proposal-cache "$root/proposal_cache_section5_v1" \
  --output "$OUTPUT" --n-detected "$N_DETECTED" \
  --injection "$INJECTION" --components $COMPONENTS \
  --h 0.00125 --previous-draws "$PREVIOUS_DRAWS" --draws "$DRAWS" \
  --proposal-candidates "$PROPOSAL_CANDIDATES" \
  --proposal-seed "$PROPOSAL_SEED" --proposal-epsilon 0.1 \
  --object-chunk 16 --atom-chunk 4096 --device cuda
