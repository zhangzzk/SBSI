#!/usr/bin/env bash
# Powered exact and autograd diagnostics for the Section 5 catalogue-prior null.

#SBATCH --job-name=sbsi_s5_exact
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

: "${OUTPUT:=$root/powered_exact_matched_v1}"
: "${EXACT_ATOMS:=4096}"
: "${EXACT_OBJECTS:=32768}"
: "${MATCHED_BANK_ATOMS:=4096}"
: "${MATCHED_BANK_OBJECTS:=8192}"
: "${AUTOGRAD_OBJECTS:=32}"
: "${H:=0.00125}"
: "${MAX_INDEPENDENT_BANKS:=3}"

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/run_section5_powered_exact.py" \
  --scene-store "$root/scene_store" \
  --measurement-model "$flow" \
  --emulator-metadata "$metadata" --emulator-model "$emulator" \
  --model-cache "$root/model_cache_section5_v1" \
  --output "$OUTPUT" \
  --exact-atoms "$EXACT_ATOMS" --exact-objects "$EXACT_OBJECTS" \
  --matched-bank-atoms "$MATCHED_BANK_ATOMS" --matched-bank-objects "$MATCHED_BANK_OBJECTS" \
  --autograd-objects "$AUTOGRAD_OBJECTS" --h "$H" --seed 5201 \
  --max-independent-banks "$MAX_INDEPENDENT_BANKS" \
  --object-chunk 32 --atom-chunk 4096 --device cuda
