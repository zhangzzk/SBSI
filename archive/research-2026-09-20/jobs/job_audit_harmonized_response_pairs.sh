#!/usr/bin/env bash
#SBATCH --job-name=sbsi_harmonized_amplitude
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/disk_response_moment_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/disk_response_moment_20260918_v1/logs/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/disk_response_moment_20260918_v1
case ${1:?smoke or run} in
  smoke) cases=(0); output=$root/fresh_amplitude_case0.json ;;
  run) cases=(0 1 40 41); output=$root/fresh_amplitude_pilot.json ;;
  *) exit 2 ;;
esac
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.audit_harmonized_response_pairs \
  --prepared-root /project/ls-gruen/users/zekang.zhang/blendemu_runs/primary_amplitude_pilot_20260918_v1/prepared \
  --cases "${cases[@]}" --output "$output"
