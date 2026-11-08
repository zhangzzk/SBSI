#!/bin/bash
# Eight independent singleton workers; preserve the original per-leg launcher.
#SBATCH --partition=cluster
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/z/Zekang.Zhang/blendemu_galsbi_resim/scripts
set -eo pipefail
mode=${1:-run}
group=${SLURM_ARRAY_TASK_ID:-0}
[[ "$group" =~ ^[0-8]$ ]] || exit 2
[[ "$mode" == run || "$mode" == smoke ]] || exit 2
logroot=/project/ls-gruen/users/zekang.zhang/sbsi_caches/alignment_axis2_constant_20261105/logs
pids=()
for slot in {0..7}; do
    task=$((8 + group * 8 + slot))
    (
        export SLURM_ARRAY_TASK_ID=$task
        if [[ "$mode" == smoke ]]; then
            export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
            source /home/z/Zekang.Zhang/blendemu/jobs/env_fs2_lsst_r.sh
            export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
            python -u -c 'import os; from mpi4py import MPI; c=MPI.COMM_WORLD; assert c.size==1 and c.rank==0; print("SMOKE",os.environ["SLURM_ARRAY_TASK_ID"],"rank",c.rank,"size",c.size,flush=True)'
        else
            exec bash /home/z/Zekang.Zhang/blendemu_galsbi_resim/jobs/job_constant_axis2.sh
        fi
    ) >"$logroot/packed_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${task}.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then status=1; fi
done
echo "PACK_COMPLETE group=$group mode=$mode status=$status"
exit "$status"
