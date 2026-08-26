#!/usr/bin/env bash
# Archived pre-Infer-V1 ConstGold preparation.
# Prepare ten disjoint five-case ConstGold +g1 image blocks for the V3.1 seed-501 screen.

#SBATCH --job-name=sbsi_v31_imgmock
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-9
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/constgold_v31_s501_g1p002_c50_99_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/constgold_v31_s501_g1p002_c50_99_v1/logs/%x_%A_%a.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
simulation_root=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant
campaign_root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/constgold_v31_s501_g1p002_c50_99_v1
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/measurement_flow_mixed_g0_g005_E_s501_swaavg.pt

first_case=$((50 + 5 * SLURM_ARRAY_TASK_ID))
last_case=$((first_case + 4))
block_tag="c${first_case}_${last_case}_n2048"
output="$campaign_root/mocks/mock_g1p002_${block_tag}"

if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi

export PYTHONPATH="$repo"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/prepare_image_closure_mock.py" \
  --simulation-root "$simulation_root" --cases "${first_case}-${last_case}" \
  --shear-label 0.02 --injected-g1 0.02 --injected-g2 0 \
  --measurement-model "$flow" --output "$output" \
  --n-objects 2048 --sample-seed "$((9501 + SLURM_ARRAY_TASK_ID))" \
  --primary-mag-min 18 --primary-mag-max 25.8 \
  --primary-re-min 0.5 --primary-re-max 1.5 --device cpu
