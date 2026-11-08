#!/bin/bash
#SBATCH --job-name=sbsi-shared
#SBATCH --partition=inter
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=08:00:00
# Several persistent-scene MH processes share one GPU: the per-step cost is mostly
# single-core Python, so one process leaves the GPU idle. Galaxies first..last are
# split evenly over P processes, each pinned to its own cpus-per-task/P cores.
# If SBSI_OBS_LIST names a file of observation indices (one per line), first..last
# are 0-based positions in that list instead of observation indices.
# Process k writes $SBSI_TEMPER_ROOT/<label>_k<k> with seeds seed+k and bank_seed+k,
# so different processes never reuse random numbers.
# Usage: SBSI_TEMPER_ROOT=... sbatch jobs/job_galsbi_shared_gpu.sh <label> <P> <first> <last> <seed> <bank_seed> [sampler args...]
set -euo pipefail
: "${SBSI_TEMPER_ROOT:?set tempering experiment root}"
label=$1 P=$2 first=$3 last=$4 seed=$5 bank_seed=$6
shift 6
cores=($(/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -c 'import os;print(*sorted(os.sched_getaffinity(0)))'))
C=$(( ${#cores[@]} / P ))
(( C >= 1 )) || { echo "need at least $P cores" >&2; exit 1; }
if [ -n "${SBSI_OBS_LIST:-}" ]; then
    ids=($(sed -n "$((first+1)),$((last+1))p" "$SBSI_OBS_LIST"))
else
    ids=($(seq $first $last))
fi
n=${#ids[@]}
mkdir -p "$SBSI_TEMPER_ROOT/logs"
pids=()
for ((k=0; k<P; k++)); do
    lo=$(( n*k/P )); hi=$(( n*(k+1)/P ))
    set=$(IFS=,; echo "${cores[*]:$((C*k)):C}")
    taskset -c "$set" bash "$SBSI_TEMPER_ROOT/source_snapshot/jobs/job_galsbi_tempered.sh" "${label}_k$k" \
        --seed $(( seed + k )) --initial-bank-seed $(( bank_seed + k )) \
        --observation-indices "${ids[@]:lo:hi-lo}" "$@" \
        > "$SBSI_TEMPER_ROOT/logs/${label}_k${k}_${SLURM_JOB_ID}.out" 2>&1 &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit $status
