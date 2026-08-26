#!/usr/bin/env bash
# Archived pre-Infer-V1 cache build.
# Build twenty neighbour-complete shard views and QMC-128 moments on exactly two A40s.

#SBATCH --job-name=sbsi_fs2_qmc128_build
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
blendemu=/home/z/Zekang.Zhang/blendemu
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/v31_fs2_default_qmc128_n100k_v1
manifest=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/fs2_prior_cases20000_20199_v1/default_prior_manifest.json
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt
emulator_metadata="$repo/models/blendemu/emulator_metadata_lsst_r_extnbr_v22.json"
emulator_model=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v22_reweighted_vector_optuna30_all40_v1/best_weighted_model.json

export PYTHONPATH="$repo:$blendemu"
export OMP_NUM_THREADS=8
mkdir -p "$root/logs" "$root/shard_caches"

build_one() {
  local gpu=$1
  local index=$2
  local tag
  tag=$(printf '%02d' "$index")
  CUDA_VISIBLE_DEVICES="$gpu" "$python" \
    "$repo/scripts/build_sharded_prior_model_qmc.py" \
    --default-prior-manifest "$manifest" --shard-index "$index" \
    --measurement-model "$flow" \
    --emulator-metadata "$emulator_metadata" --emulator-model "$emulator_model" \
    --qmc-samples 128 --qmc-row-chunk 8192 --qmc-seed 8201 \
    --output "$root/shard_caches/shard_$tag" \
    >"$root/logs/shard_$tag.out" 2>"$root/logs/shard_$tag.err"
}

for wave in $(seq 0 9); do
  first=$((2 * wave))
  second=$((first + 1))
  build_one 0 "$first" &
  first_pid=$!
  build_one 1 "$second" &
  second_pid=$!
  wait "$first_pid"
  wait "$second_pid"
done

echo "all twenty shard caches complete"
