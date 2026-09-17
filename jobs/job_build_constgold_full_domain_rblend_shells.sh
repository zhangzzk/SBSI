#!/usr/bin/env bash
# R_blend for the full-domain anchors, resolved by neighbour separation.
#
# The 2026-09-17 split measured the blending term as -15.41% +- 0.53 too large
# while the emulator's own metadata reports a per-pair development slope of
# 1.0406 +- 0.0035 -- a ~4% UNDER-prediction.  Opposite signs, so something
# happens between the single pair and the sum over pairs.
#
# Profiling the per-pair predictions against separation on case 40 shows what:
# R_blend is a signed sum over the 20 nearest neighbours inside r_max = 10
# ARCSEC, the cap binds for nearly every primary (mean 16 pairs, 33% at exactly
# the maximum), and only 4.6% of the total comes from inside 1.5 arcsec while
# 62% comes from beyond 3 arcsec -- 15 pixels, against galaxies of order half an
# arcsecond under a 0.73 arcsec PSF.  At fixed flux ratio the predicted response
# falls monotonically and then turns significantly NEGATIVE beyond 5 arcsec,
# where the physical effect should be zero.
#
# Truncating the neighbour list at the separation that makes m small would be a
# fitted offset, and a worthless one: 5 arcsec gives m = +2.10%, 6 arcsec gives
# -0.66%.  So instead this emits the per-shell sums and lets ConstGold measure
# each shell's own fractional error jointly.  If the outer shells come back
# consistent with contributing nothing real, that is a measurement.
#
# Shell edges 2/4/6 arcsec are round numbers spanning the aperture, fixed before
# any m was read from them.  The share-at-4-arcsec column is the binning axis:
# binning on the total amplitude leaves the four shells nearly parallel across
# bins (scaled condition number 63, against 29 for the share), which would leave
# the split between them unconstrained however small its errors looked.
#
# R_blend, R_blend_abs and blend_pairs are byte-for-byte what the existing
# lookup carries -- same anchors, same truth, same emulator, same measurement
# model -- so the new blocks can be differenced against the old ones as a check
# that only new columns were added.
#
# CPU only, so it does not consume the owner's two-GPU budget.

#SBATCH --job-name=sbsi_cgshell_rb
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-9%5
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/constgold_rblend_shells_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/constgold_rblend_shells_v1/logs/%x_%A_%a.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/perleg-response
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
blend=/project/ls-gruen/users/zekang.zhang/blendemu_runs
refine=$cache/full_domain_true_mag26_guard_refine_v1
domain=$cache/full_domain_true_mag26_guard_v1/domain
root=$refine/constgold_rblend_shells_v1
anchor=$domain/flow/g0/case{case:03d}.npz
truth=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather
#  The same measurement model the existing lookup was built with, so R_blend
#  reproduces bit for bit.  It is validated but unused: `load_emulator` calls
#  `BlendingPredictor.load(..., load_self=False)` and never passes the flow, as
#  verified on 9,802,054 rows where swapping it changed nothing.
loader=$cache/fixed_g0_m258_r060_v2/flow/nll/selected.pt
model_root=$blend/fixed_g0_m258_r060_v2/models
model=$model_root/regression_model_lsst_r_fixed_g0_m258_r060_v2.json
metadata=$model_root/emulator_metadata_lsst_r_fixed_g0_m258_r060_v2.json

[ -f "$loader" ] || { echo "missing flow: $loader" >&2; exit 3; }

first=$((40 + 10 * SLURM_ARRAY_TASK_ID))
last=$((first + 9))
output=$root/rblend/blocks/cases${first}_${last}.feather
if [[ -e "$output" || -e "${output%.feather}.json" ]]; then
  echo "SKIP block already present: $output"
  exit 0
fi
mkdir -p "$root/logs" "$root/rblend/blocks"
mapfile -t cases < <(seq "$first" "$last")

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$repo:/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

"$python" -u -m scripts.build_constgold_fixed_g0_blend_lookup \
  --cases "${cases[@]}" \
  --input-pattern "$truth" \
  --anchor-pattern "$anchor" \
  --measurement-model "$loader" \
  --emulator-model "$model" \
  --emulator-metadata "$metadata" \
  --shell-edge 2 --shell-edge 4 --shell-edge 6 \
  --share-edge 4 \
  --output "$output" \
  --device cpu

echo "CONSTGOLD_SHELL_RBLEND_BLOCK_DONE cases=$first-$last output=$output"
