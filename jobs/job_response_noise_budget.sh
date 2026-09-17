#!/usr/bin/env bash
# Isolate the flow's Monte Carlo draw noise in the per-leg response.
#
# Every per-leg measurement so far shared the hard-coded EVALUATION_SEED, which
# makes checkpoint comparisons clean common-random-number differences but means
# the draw noise has never been separated from anything.  The simulation side of
# the ratio does not depend on that seed at all, so repeating one fixed
# checkpoint at three seeds and holding everything else fixed leaves the flow's
# draws as the only thing that changed.  The run also writes the per-case
# group-spread diagnostic, which estimates the same quantity from the four
# independent latent replicas within a single measurement.

#SBATCH --job-name=sbsi_noise_budget
#SBATCH --partition=cip,inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1/logs/%x_%j.err

set -euo pipefail
repo=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/perleg-response
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
domain_root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_guard_v1/domain
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/full_domain_true_mag26_band_guard_v1
flow=$root/weight_0.1/selected.pt
output=$root/noise_budget

if [ ! -f "$flow" ]; then
  echo "missing checkpoint: $flow" >&2
  exit 1
fi

mkdir -p "$output" "$root/logs"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$repo"
nvidia-smi -L
"$python" -c "import sbsi; print('sbsi from', sbsi.__file__)"

# The first seed reproduces the published measurement, so any difference
# between it and the stored result is a bug rather than noise.
for seed in 733000000001 811000000001 907000000001; do
  target=$output/seed_$seed.json
  if [ -f "$target" ]; then
    echo "SKIP $target already present"
    continue
  fi
  "$python" -u -m scripts.evaluate_per_leg_hard_cut_response \
    --domain-root "$domain_root" \
    --flow "$flow" \
    --output "$target" \
    --evaluation-seed "$seed" \
    --draws 16 --groups 4 --batch-size 2048 --device cuda
done

echo "RESPONSE_NOISE_BUDGET_COMPLETE output=$output"
