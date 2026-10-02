#!/bin/bash
#SBATCH --job-name=sbsi-throughput
#SBATCH --partition=inter
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
: "${SBSI_SPEED_ROOT:?set bounded throughput root}"
: "${SBSI_SPEED_STAGE:?set stage}"
: "${SBSI_CACHE_DIR:?set artifact root}"
: "${BLENDEMU_RUNS_DIR:?set artifact root}"
cd "$SBSI_SPEED_ROOT/source_snapshot"
sha256sum --check SHA256SUMS --quiet
SBSI_SPEED_PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
"$SBSI_SPEED_PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py --output "$SBSI_SPEED_ROOT/logs/resources_${SLURM_JOB_ID}.json"
if [[ "$SBSI_SPEED_STAGE" == report ]]; then
    SBSI_SPEED_SCRIPT=scripts/report_galsbi_ensemble.py
    SBSI_SPEED_ARGS=(--root "$SBSI_SPEED_ROOT")
else
    SBSI_SPEED_SCRIPT=scripts/run_galsbi_throughput.py
    SBSI_SPEED_ARGS=(--root "$SBSI_SPEED_ROOT" --stage "$SBSI_SPEED_STAGE" --index "${SLURM_ARRAY_TASK_ID:-0}")
fi
"$SBSI_SPEED_PYTHON" -u -c 'import pathlib,runpy,sys;sys.path[:0]=[".","scripts"];import sbsi;assert pathlib.Path(sbsi.__file__).resolve().parent.parent==pathlib.Path.cwd();script=sys.argv.pop(1);runpy.run_path(script,run_name="__main__")' "$SBSI_SPEED_SCRIPT" "${SBSI_SPEED_ARGS[@]}"
