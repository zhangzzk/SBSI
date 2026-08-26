#!/usr/bin/env bash
# Matched-population image-mock pilot from existing FS2 constant-shear renders.

#SBATCH --job-name=sbsi_imgmock
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/image_closure_fs2_constgold_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/image_closure_fs2_constgold_v1/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
flow=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation/measurement_flow_g0_ngmix_ablate_s2c_lt500_v22_s501_swaavg.pt
: "${SIMULATION_ROOT:=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant}"
: "${CASES:=40-44}"
: "${SHEAR_LABEL:=0.02}"
: "${INJECTED_G1:=0.02}"
: "${INJECTED_G2:=0}"
: "${N_OBJECTS:=2048}"
: "${SAMPLE_SEED:=9401}"
: "${OUTPUT:=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/image_closure_fs2_constgold_v1/mock_g1p002_c40_44_n2048}"

if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite $OUTPUT" >&2
  exit 2
fi

export PYTHONPATH="$repo"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$python" "$repo/scripts/prepare_image_closure_mock.py" \
  --simulation-root "$SIMULATION_ROOT" --cases "$CASES" \
  --shear-label "$SHEAR_LABEL" \
  --injected-g1 "$INJECTED_G1" --injected-g2 "$INJECTED_G2" \
  --measurement-model "$flow" --output "$OUTPUT" \
  --n-objects "$N_OBJECTS" --sample-seed "$SAMPLE_SEED" \
  --primary-mag-min 18 --primary-mag-max 25.8 \
  --primary-re-min 0.5 --primary-re-max 1.5 --device cpu
