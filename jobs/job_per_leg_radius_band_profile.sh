#!/usr/bin/env bash
# Decompose the per-leg hard-cut response error into disjoint radius bands.
#
# The cumulative guards report the mean response of everything above a
# threshold, so an error concentrated just above the cut is diluted by the
# well-resolved majority.  Disjoint bands remove that dilution and show which
# sizes actually carry the residual bias.

#SBATCH --job-name=sbsi_band_profile
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/per_leg_band_profile_c40_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1/per_leg_band_profile_c40_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/perleg-response
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
domain_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/domain
refine_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_refine_v1
output=$refine_root/per_leg_band_profile_c40_v1
edges="2.2 2.6 2.8 3.0 3.2 3.5 4.0 5.0 7.0"

mkdir -p "$output/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L
"$python" -c "import sbsi; print('sbsi from', sbsi.__file__)"

# Deployment population: the same magnitude limit as the realistic cut.
# Each variant is skipped when its result already exists, so the job is safe
# to resubmit after a partial run without overwriting a completed result.
if [ -f "$output/bands_magnitude_lt_25.8.json" ]; then
  echo "SKIP bands_magnitude_lt_25.8.json already present"
else
  "$python" -u -m scripts.evaluate_per_leg_hard_cut_response \
    --domain-root "$domain_root" \
    --flow "$refine_root/flow/selected.pt" \
    --output "$output/bands_magnitude_lt_25.8.json" \
    --radius-band-edges $edges --band-magnitude-max 25.8 \
    --draws 16 --groups 4 --batch-size 2048 --device cuda
fi

# No magnitude limit, to separate a size effect from a depth effect.
if [ -f "$output/bands_all_magnitudes.json" ]; then
  echo "SKIP bands_all_magnitudes.json already present"
else
  "$python" -u -m scripts.evaluate_per_leg_hard_cut_response \
    --domain-root "$domain_root" \
    --flow "$refine_root/flow/selected.pt" \
    --output "$output/bands_all_magnitudes.json" \
    --radius-band-edges $edges \
    --draws 16 --groups 4 --batch-size 2048 --device cuda
fi

echo "BAND_PROFILE_COMPLETE output=$output"
