#!/bin/bash
#SBATCH --job-name=sbsi-domain-observed
#SBATCH --partition=cip
#SBATCH --nodelist=cip-cl-nv01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/v36_domain_observed_%j.out
set -euo pipefail
cd /home/z/Zekang.Zhang/SBSI
export PYTHONPATH=/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
"$PYTHON" -m pytest tests -q
mapfile -t cases < <(seq 40 119)
"$PYTHON" -u scripts/audit_v36_domain_response.py \
  --catalogues /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant \
  --parent /project/ls-gruen/users/zekang.zhang/sbsi_caches/constgold_truth_parent_20260917_v1/parent_v2 \
  --reference /project/ls-gruen/users/zekang.zhang/sbsi_caches/scene_classifier_geometry_20260918_v1/smooth_joint_main.json \
  --output "/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_domain_recovery_20260924_v1/observed_${SLURM_JOB_ID}" \
  --cases "${cases[@]}" --observed-only
