#!/usr/bin/env bash
# Full tests followed by shared matched-cell targets and case covariances.
#SBATCH --job-name=sbsi_gridcov_prep
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
set -euo pipefail
: "${GRID_TRIAL_ROOT:?Set a new external trial root}"
repo=/home/z/Zekang.Zhang/SBSI
base=/project/ls-gruen/users/zekang.zhang/sbsi_caches/plain_complete_flow_response_re037_v1
python=$base/distribution_response_v1/audit_env/bin/python
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$repo"
"$python" "$GRID_TRIAL_ROOT/verify_sources.py"
"$python" -m pytest tests -q
"$python" -u -m scripts.train_flow_grid_response prepare \
  --root "$GRID_TRIAL_ROOT/prepared" \
  --parent "$base/r100_t100_v1/measurement_flow_plain_direct_r100_t100_d16_re037_s501_swaavg.pt" \
  --grid "$base/response_target_mixed_train160_6x6x5_matrix_re037.npz" \
  --g0 /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_crowd_conc_g0.0_train_full.feather \
  --g1 /project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/det_meas_crowd_conc_g0.05_all200.feather
"$python" "$GRID_TRIAL_ROOT/verify_sources.py"
