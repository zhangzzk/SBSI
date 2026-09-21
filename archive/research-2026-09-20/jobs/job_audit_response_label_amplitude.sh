#!/usr/bin/env bash
#SBATCH --job-name=sbsi_label_amplitude
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%A_%a.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1/logs/%x_%A_%a.err
set -euo pipefail
root=/project/ls-gruen/users/zekang.zhang/sbsi_caches/output_conditioned_response_20260917_v1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/z/Zekang.Zhang/SBSI
mode=${1:?smoke, run or summarize}
if [[ $mode == summarize ]]; then
  exec /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.audit_response_label_amplitude summarize \
    --parts "$root/amplitude_parts" --output "$root/label_amplitude.json"
elif [[ $mode == smoke ]]; then
  first=0; stop=2; output=$root/label_amplitude_smoke.json
elif [[ $mode == run ]]; then
  first=$((10 * SLURM_ARRAY_TASK_ID)); stop=$((first+10)); output=$root/amplitude_parts/block${SLURM_ARRAY_TASK_ID}.json
else
  exit 2
fi
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -m scripts.audit_response_label_amplitude compute \
  --source /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876 \
  --prepared /project/ls-gruen/users/zekang.zhang/blendemu_runs/output_conditioned_response_20260917_v1/prepared \
  --case-start "$first" --case-stop "$stop" --output "$output"
