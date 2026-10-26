#!/bin/bash
#SBATCH --job-name=sbsi-bright-mh
#SBATCH --partition=inter
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
: "${SBSI_BRIGHT_ROOT:?set bright experiment root}"
: "${SBSI_BRIGHT_STAGE:?set stage}"
: "${SBSI_CACHE_DIR:?set artifact root}"
: "${BLENDEMU_RUNS_DIR:?set artifact root}"
cd "$SBSI_BRIGHT_ROOT/source_snapshot"
sha256sum --check SHA256SUMS --quiet
SBSI_BRIGHT_PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
"$SBSI_BRIGHT_PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py --output "$SBSI_BRIGHT_ROOT/logs/resources_${SLURM_JOB_ID}.json"
if [[ "$SBSI_BRIGHT_STAGE" == report ]]; then
    SBSI_BRIGHT_SCRIPT=scripts/report_galsbi_bright.py
    SBSI_BRIGHT_ARGS=(--root "$SBSI_BRIGHT_ROOT")
else
    SBSI_BRIGHT_SCRIPT=scripts/run_galsbi_bright.py
    SBSI_BRIGHT_ARGS=(--root "$SBSI_BRIGHT_ROOT" --stage "$SBSI_BRIGHT_STAGE" --replica "${SLURM_ARRAY_TASK_ID:-0}")
fi
"$SBSI_BRIGHT_PYTHON" -u -c 'import pathlib,runpy,sys;sys.path[:0]=[".","scripts"];import sbsi;assert pathlib.Path(sbsi.__file__).resolve().parent.parent==pathlib.Path.cwd();script=sys.argv.pop(1);runpy.run_path(script,run_name="__main__")' "$SBSI_BRIGHT_SCRIPT" "${SBSI_BRIGHT_ARGS[@]}"
