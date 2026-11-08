#!/bin/bash
#SBATCH --job-name=sbsi-overnight-report
#SBATCH --partition=cip
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
SBSI_OVERNIGHT_PYTHON=${SBSI_OVERNIGHT_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
: "${SBSI_OVERNIGHT_ROOT:?set study root}"
: "${SBSI_OVERNIGHT_SNAPSHOT:?set frozen source checkout}"
cd "$SBSI_OVERNIGHT_SNAPSHOT"
sha256sum --check SHA256SUMS --quiet
SBSI_MH_SEED_BASE=${SBSI_MH_SEED_BASE:-2026100101}
SBSI_MH_REPLICAS=${SBSI_MH_REPLICAS:-2}
SBSI_MH_REPORT_RUNS=()
for ((i=0; i<SBSI_MH_REPLICAS; i++)); do
    SBSI_MH_REPORT_RUNS+=("$SBSI_OVERNIGHT_ROOT/seed$((SBSI_MH_SEED_BASE+i))")
done
"$SBSI_OVERNIGHT_PYTHON" -u -c 'import runpy, sys; sys.path[:0]=[".", "scripts"]; runpy.run_path("scripts/report_galsbi_sampler_runs.py", run_name="__main__")' \
    "${SBSI_MH_REPORT_RUNS[@]}" \
    --output "$SBSI_OVERNIGHT_ROOT/report" --prefixes 65536 131072 262144 524288 1048576
