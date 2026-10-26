#!/usr/bin/env bash
# Restore replica concurrency only after the other GPU array has completed.
#SBATCH --job-name=sbsi_restore_replica_throttle
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/flow_shape_augmentation_20260918_v1/logs/%x_%j.err
set -euo pipefail
replica_array_id=${1:?replica array job ID required}
[[ "$replica_array_id" =~ ^[0-9]+$ ]]
running_jobs=$(squeue -h -u "$USER" -t RUNNING,CONFIGURING -o '%F|%b')
while IFS='|' read -r job_array_id requested_gres; do
    if [[ "$requested_gres" == *gpu* && "$job_array_id" != "$replica_array_id" ]]; then
        echo "OTHER_GPU_JOB_PRESENT array=$job_array_id gres=$requested_gres" >&2
        exit 1
    fi
done <<< "$running_jobs"
scontrol update JobId="$replica_array_id" ArrayTaskThrottle=2
scontrol show job "$replica_array_id"
