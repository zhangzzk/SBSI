#!/usr/bin/env bash
# Archived Infer V1 sampling diagnostic job.
# Audit Gaussian-ranked candidate capture and extend the deep reference if needed.

#SBATCH --job-name=sbsi_fs2_kcapture
#SBATCH --partition=inter
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_km_n20k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_km_n20k_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
source_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_km_n20k_v1
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
metadata="$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json"
emulator=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

common=(
  --scene-store "$source_root/compact_global/scene_store"
  --measurement-model "$flow"
  --emulator-metadata "$metadata"
  --emulator-model "$emulator"
  --model-cache "$source_root/compact_global/model_cache"
  --proposal-cache "$source_root/compact_global/proposal_cache"
  --mock-input "$source_root/prepare/mock"
  --n-objects 1024
  --center 0.014022970805913502 0.0023700238746689867
  --h 0.001 --full-stencil
  --extension-trigger-prefix 32768 --extension-min-q01 0.999
  --object-chunk 16 --atom-chunk 4096 --compile-flow --device cuda
)

"$python" "$repo/scripts/diagnose_deep_candidate_capture.py" \
  "${common[@]}" \
  --reference-candidates 65536 \
  --candidate-prefixes 8192 16384 32768 65536 \
  --output "$root/candidate_capture_kref65536"

extend=$("$python" -c \
  'import json,sys; print(int(json.load(open(sys.argv[1]))["extension_rule"]["recommend_extend_reference"]))' \
  "$root/candidate_capture_kref65536/result.json")
if [[ "$extend" == 1 ]]; then
  "$python" "$repo/scripts/diagnose_deep_candidate_capture.py" \
    "${common[@]}" \
    --reference-candidates 131072 \
    --candidate-prefixes 8192 16384 32768 65536 131072 \
    --output "$root/candidate_capture_kref131072"
fi

echo "candidate-capture audit complete; extended=$extend"
