#!/usr/bin/env bash
#SBATCH --job-name=sbsi_disk_density
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-16gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/disk_response_moment_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/disk_response_moment_20260918_v1/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI${PYTHONPATH:+:$PYTHONPATH}"
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/disk_response_moment_20260918_v1
cd /home/z/Zekang.Zhang/SBSI
case ${1:?smoke or run} in
  smoke) cases=(40 41); output=$root/density_smoke.json ;;
  run) mapfile -t cases < <(seq 40 119); output=$root/density_full.json ;;
  *) exit 2 ;;
esac
script=${2:-/home/z/Zekang.Zhang/SBSI/scripts/audit_disk_candidate_density.py}
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u "$script" \
  --cases "${cases[@]}" --output "$output"
