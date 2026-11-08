#!/bin/bash
#SBATCH --job-name=sbsi-population
#SBATCH --partition=inter
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
: "${SBSI_POP_ROOT:?set population study root}"
: "${SBSI_CACHE_DIR:?set artifact root}"
: "${BLENDEMU_RUNS_DIR:?set artifact root}"
cd "$SBSI_POP_ROOT/source_snapshot"
sha256sum --check SHA256SUMS --quiet
SBSI_POP_PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
SBSI_POP_STAGE=${SBSI_POP_STAGE:-sample}
SBSI_POP_INDEX=${SLURM_ARRAY_TASK_ID:-0}
SBSI_POP_ARGS=()
case "$SBSI_POP_STAGE" in
    prepare)
        SBSI_POP_SCRIPT=scripts/prepare_galsbi_population.py
        SBSI_POP_ARGS=(--reference /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_summary_mh_20260930_v2 --output "$SBSI_POP_ROOT/reference" --observations 20000 --seed 2026100301)
        ;;
    sample|smoke)
        SBSI_POP_SCRIPT=scripts/run_galsbi_population_shard.py
        SBSI_POP_ARGS=(--root "$SBSI_POP_ROOT" --shard "$SBSI_POP_INDEX")
        if [[ "$SBSI_POP_STAGE" == smoke ]]; then SBSI_POP_ARGS+=(--smoke); fi
        ;;
    report|combine)
        SBSI_POP_SCRIPT=scripts/report_galsbi_population.py
        SBSI_POP_ARGS=(--root "$SBSI_POP_ROOT")
        if [[ "$SBSI_POP_STAGE" == report ]]; then SBSI_POP_ARGS+=(--shard "$SBSI_POP_INDEX"); fi
        ;;
    *) exit 2 ;;
esac
"$SBSI_POP_PYTHON" -u -c 'import pathlib,runpy,sys;sys.path[:0]=[".","scripts"];import sbsi;assert pathlib.Path(sbsi.__file__).resolve().parent.parent==pathlib.Path.cwd();script=sys.argv.pop(1);runpy.run_path(script,run_name="__main__")' "$SBSI_POP_SCRIPT" "${SBSI_POP_ARGS[@]}"
