#!/bin/bash
#SBATCH -p cluster
#SBATCH -c 1
#SBATCH --mem 24G
#SBATCH -t 04:00:00
#SBATCH -J galsbi-scene
#SBATCH -o /project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/logs/scene_%A_%a.log
# One GalSBI prior scene per array task: case = CASE0 + task id. Usage:
#   sbatch --array=0-9 --export=ALL,CASE0=0,SCENE_SEED=20261101,SCENE_OUT=<dir> jobs/job_write_galsbi_scenes.sh
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
: "${CASE0:?}" "${SCENE_SEED:?}" "${SCENE_OUT:?}"
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe
CASE=$((CASE0 + SLURM_ARRAY_TASK_ID))
/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -c 'import pathlib,runpy,sys;sys.path[:0]=[".","scripts"];import sbsi;assert pathlib.Path(sbsi.__file__).resolve().parent.parent==pathlib.Path.cwd();script=sys.argv.pop(1);runpy.run_path(script,run_name="__main__")' \
  scripts/write_galsbi_scenes.py --cases "$CASE" "$((CASE + 1))" --seed "$SCENE_SEED" \
  --lsst-table /project/ls-gruen/users/zekang.zhang/lsst_filters/LSST_template_integrals_ebv0.h5 --out "$SCENE_OUT"
