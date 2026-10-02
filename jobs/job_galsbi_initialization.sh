#!/bin/bash
#SBATCH --job-name=sbsi-init
#SBATCH --partition=cip
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:25:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
: "${SBSI_INIT_ROOT:?set study root}"
: "${SBSI_CACHE_DIR:?set artifact root}"
: "${BLENDEMU_RUNS_DIR:?set response artifact root}"
SBSI_INIT_PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
cd "$SBSI_INIT_ROOT/source_snapshot"
sha256sum --check SHA256SUMS --quiet
SBSI_INIT_INDICES=("${SLURM_ARRAY_TASK_ID:-0}")
SBSI_INIT_DEVICE=cuda
SBSI_INIT_STEPS=16384
SBSI_INIT_WARMUP=4096
SBSI_INIT_STAGE=pilot
if [[ ${SBSI_INIT_SMOKE:-0} == 1 ]]; then
    SBSI_INIT_INDICES=(0 1 2 3)
    SBSI_INIT_DEVICE=cpu
    SBSI_INIT_STEPS=64
    SBSI_INIT_WARMUP=32
    SBSI_INIT_STAGE=smoke
fi
for SBSI_INIT_INDEX in "${SBSI_INIT_INDICES[@]}"; do
    # Counterbalance arm order across the two seed pairs.
    SBSI_INIT_ARMS=(prior-bank measurement measurement prior-bank)
    SBSI_INIT_REPLICAS=(0 0 1 1)
    SBSI_INIT_ARM=${SBSI_INIT_ARMS[$SBSI_INIT_INDEX]}
    SBSI_INIT_REPLICA=${SBSI_INIT_REPLICAS[$SBSI_INIT_INDEX]}
    SBSI_INIT_SEED=$((2026100141 + SBSI_INIT_REPLICA))
    SBSI_INIT_BANK_SEED=$((2026100241 + SBSI_INIT_REPLICA))
    SBSI_INIT_OUTPUT="$SBSI_INIT_ROOT/$SBSI_INIT_STAGE/$SBSI_INIT_ARM/seed$SBSI_INIT_SEED"
    mkdir -p "$SBSI_INIT_OUTPUT"
    trap 'SBSI_INIT_EXIT=$?; printf "{\"exit_code\":%d,\"job_id\":\"%s\"}\n" "$SBSI_INIT_EXIT" "$SLURM_JOB_ID" > "$SBSI_INIT_OUTPUT/job_exit.json"' EXIT
    "$SBSI_INIT_PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py --output "$SBSI_INIT_OUTPUT/resources.json"
    "$SBSI_INIT_PYTHON" -u -c 'import pathlib,runpy,sys;sys.path[:0]=[".","scripts"];import sbsi;assert pathlib.Path(sbsi.__file__).resolve().parent.parent==pathlib.Path.cwd();runpy.run_path("scripts/check_galsbi_local_sampler.py",run_name="__main__")' \
        --reference /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_summary_mh_20260930_v2 \
        --output "$SBSI_INIT_OUTPUT" --device "$SBSI_INIT_DEVICE" \
        --seed "$SBSI_INIT_SEED" --initial-bank-seed "$SBSI_INIT_BANK_SEED" --initialization "$SBSI_INIT_ARM" \
        --steps "$SBSI_INIT_STEPS" --warmup "$SBSI_INIT_WARMUP" --observation-indices 0 1 2 3 \
        --environment-update persistent --environments 1 --marginalize-type \
        --scale-adaptation per-chain --sed-jump --joint-morphology
    printf '{"exit_code":0,"job_id":"%s"}\n' "$SLURM_JOB_ID" > "$SBSI_INIT_OUTPUT/job_exit.json"
done
