#!/usr/bin/env bash
# Build fixed-g0 R_blend blocks on constgold cases 40-89 for one arm of the
# bright-neighbour rejection A/B.  Pass --export=ALL,VERSION=v2_relaxed|v2_baseline

#SBATCH --job-name=sbsi_cgfg0_rb_ab
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_blend_lookup_abs_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/constgold_blend_lookup_abs_v1/logs/%x_%A_%a.err

set -euo pipefail
: "${VERSION:?set VERSION=v2_relaxed or v2_baseline}"
repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
blend=/project/ls-gruen/users/zekang.zhang/blendemu_runs
root=$cache/fixed_g0_m258_r060_v2/constgold_blend_lookup_abs_v1
anchor=$cache/fixed_g0_m258_r060_v2/domain/flow/g0/case{case:03d}.npz
truth=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather
loader=$cache/fixed_g0_m258_r060_v2/flow/nll/selected.pt

first=$((40 + 10 * SLURM_ARRAY_TASK_ID))
last=$((first + 9))
mkdir -p "$root/logs" "$root/rblend_${VERSION}/blocks"
mapfile -t cases < <(seq "$first" "$last")

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo:/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

# v2_relaxed -> models dir fixed_g0_m258_r060_v2_relaxed, tag ..._v2_relaxed
arm=${VERSION#v2_}
model_root=$blend/fixed_g0_m258_r060_v2_${arm}/models
model=$model_root/regression_model_lsst_r_fixed_g0_m258_r060_v2_${arm}.json
metadata=$model_root/emulator_metadata_lsst_r_fixed_g0_m258_r060_v2_${arm}.json
for f in "$model" "$metadata"; do
  [ -f "$f" ] || { echo "missing input: $f" >&2; exit 3; }
done

output=$root/rblend_${VERSION}/blocks/cases${first}_${last}.feather
"$python" -u -m scripts.build_constgold_fixed_g0_blend_lookup \
  --cases "${cases[@]}" \
  --input-pattern "$truth" \
  --anchor-pattern "$anchor" \
  --measurement-model "$loader" \
  --emulator-model "$model" \
  --emulator-metadata "$metadata" \
  --output "$output" \
  --device cpu

echo "CONSTGOLD_RB_AB_BLOCK_DONE version=$VERSION cases=$first-$last"
