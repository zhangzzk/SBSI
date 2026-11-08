#!/bin/bash
#SBATCH --job-name=sbsi-overnight
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40-8gb:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
SBSI_OVERNIGHT_PYTHON=${SBSI_OVERNIGHT_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
: "${SBSI_OVERNIGHT_ROOT:?set fresh study root}"
: "${SBSI_OVERNIGHT_SNAPSHOT:?set frozen source checkout}"
: "${SBSI_OVERNIGHT_REFERENCE:?set validated summary reference}"
: "${SBSI_CACHE_DIR:?set model artifact root}"
: "${BLENDEMU_RUNS_DIR:?set response artifact root}"
SBSI_MH_SEED_BASE=${SBSI_MH_SEED_BASE:-2026100101}
SBSI_OVERNIGHT_SEED=$((SBSI_MH_SEED_BASE + SLURM_ARRAY_TASK_ID))
SBSI_OVERNIGHT_OUTPUT="$SBSI_OVERNIGHT_ROOT/seed$SBSI_OVERNIGHT_SEED"
SBSI_MH_STEPS=${SBSI_MH_STEPS:-262144}
SBSI_MH_WARMUP=${SBSI_MH_WARMUP:-32768}
SBSI_MH_EXTRA_ARGS=()
if [[ -n ${SBSI_MH_BANK_SEED_BASE:-} ]]; then
    SBSI_MH_EXTRA_ARGS+=(--initial-bank-seed "$((SBSI_MH_BANK_SEED_BASE + SLURM_ARRAY_TASK_ID))")
fi
if [[ ${SBSI_MH_POOLED:-0} == 1 ]]; then
    SBSI_MH_EXTRA_ARGS+=(--pooled-morphology)
fi
if [[ ${SBSI_MH_JOINT_ENVIRONMENT:-0} == 1 ]]; then
    SBSI_MH_EXTRA_ARGS+=(--joint-environment)
fi
mkdir -p "$SBSI_OVERNIGHT_OUTPUT"
trap 'SBSI_OVERNIGHT_EXIT=$?; printf "{\"exit_code\":%d,\"job_id\":\"%s\"}\n" "$SBSI_OVERNIGHT_EXIT" "$SLURM_JOB_ID" > "$SBSI_OVERNIGHT_OUTPUT/job_exit.json"' EXIT
cd "$SBSI_OVERNIGHT_SNAPSHOT"
sha256sum --check SHA256SUMS --quiet
"$SBSI_OVERNIGHT_PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py --output "$SBSI_OVERNIGHT_OUTPUT/resources.json"
# Prior/model imports must resolve to the frozen checkout, not the editable
# installation. The scientific runner also checks every model artifact hash.
"$SBSI_OVERNIGHT_PYTHON" -u -c 'import pathlib, runpy, sys; sys.path[:0]=[".", "scripts"]; import sbsi; assert pathlib.Path(sbsi.__file__).resolve().parent.parent == pathlib.Path.cwd(); print("FROZEN_CHECKOUT", pathlib.Path.cwd(), flush=True); runpy.run_path("scripts/check_galsbi_local_sampler.py", run_name="__main__")' \
    --reference "$SBSI_OVERNIGHT_REFERENCE" --output "$SBSI_OVERNIGHT_OUTPUT" \
    --seed "$SBSI_OVERNIGHT_SEED" --steps "$SBSI_MH_STEPS" --warmup "$SBSI_MH_WARMUP" \
    --observation-indices 0 1 2 3 --environment-update persistent --environments 1 \
    --marginalize-type --scale-adaptation per-chain --sed-jump --joint-morphology "${SBSI_MH_EXTRA_ARGS[@]}"
