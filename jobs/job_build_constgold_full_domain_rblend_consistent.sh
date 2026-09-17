#!/usr/bin/env bash
# R_blend for the full-domain anchors, built with the SAME flow used for R_flow.
#
# The deployed ConstGold model is R = R_flow + R_blend, but the two terms are
# currently computed with different measurement models.  R_flow comes from
# `full_domain_true_mag26_guard_refine_v1/flow/selected.pt`, while
# job_build_constgold_full_domain_rblend.sh builds R_blend with
# `fixed_g0_m258_r060_v2/flow/nll/selected.pt` -- a flow trained on a different,
# measurement-cut domain (mag<25.8, r>0.60) and trained on likelihood alone,
# with no response supervision of any kind.
#
# That matters because `predict_response` propagates the emulator's per-pair
# effect through the measurement model, so the flow's own response enters
# R_blend.  Supplying a flow whose response was never supervised, on objects
# outside the domain it was fitted to, is a plausible source of a large and
# roughly uniform fractional error in R_blend -- and the 2026-09-17 split
# measured exactly that, -15.4% +- 0.53, sitting at -11.5% to -16.4% across the
# three bins where R_blend is large enough to be read.
#
# This run changes the measurement model and nothing else: identical anchors,
# identical truth, identical emulator model and metadata, identical cases and
# block boundaries.  Differencing the two lookups on the same anchor keys
# measures how much of R_blend is the flow choice rather than the emulator.
#
# CPU only, like the job it mirrors, so it does not consume the two-GPU budget.

#SBATCH --job-name=sbsi_cgcons_rb
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-9%5
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/constgold_rblend_consistent_flow_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/constgold_rblend_consistent_flow_v1/logs/%x_%A_%a.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/perleg-response
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
blend=/project/ls-gruen/users/zekang.zhang/blendemu_runs
refine=$cache/full_domain_true_mag26_guard_refine_v1
domain=$cache/full_domain_true_mag26_guard_v1/domain
root=$refine/constgold_rblend_consistent_flow_v1
anchor=$domain/flow/g0/case{case:03d}.npz
truth=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/case{case}_0.02/real0/catalogues/input/gals_info_tile180.0_-0.5.feather
#  The one changed argument: the flow that also produces R_flow.
loader=$refine/flow/selected.pt
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
  --output "$output" \
  --device cpu

echo "CONSTGOLD_CONSISTENT_RBLEND_BLOCK_DONE cases=$first-$last output=$output"
