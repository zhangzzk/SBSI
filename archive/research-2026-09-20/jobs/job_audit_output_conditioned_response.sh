#!/usr/bin/env bash
#SBATCH --job-name=sbsi_output_response
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%A_%a.err
set -euo pipefail
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1
manifest=/project/ls-gruen/users/zekang.zhang/sbsi_caches/fixed_g0_m258_r060_v2/anchorblend_g002_measured_cuts_v1/manifest.json
models=/project/ls-gruen/users/zekang.zhang/blendemu_runs/output_conditioned_response_20260917_v1/models_cpu
export PYTHONPATH="/home/z/Zekang.Zhang/blendemu${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
case ${1:?smoke, run or summarize} in
  smoke) first=400; stop=402; output=$root/smoke.json ;;
  run) first=$((400 + 25 * SLURM_ARRAY_TASK_ID)); stop=$((first + 25)); output=$root/parts/block${SLURM_ARRAY_TASK_ID}.json ;;
  summarize)
    exec /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.audit_neighbour_emulator_amplitude summarize --parts "$root/parts" --output "$root/result.json"
    ;;
  *) exit 2 ;;
esac
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.audit_neighbour_emulator_amplitude compute \
  --case-start "$first" --case-stop "$stop" --manifest "$manifest" --output "$output" --candidate-root "$models"
