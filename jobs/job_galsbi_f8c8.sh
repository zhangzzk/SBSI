#!/bin/bash
#SBATCH -J galsbi-f8c8
#SBATCH -o /project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_f8c8_20261102/logs/%x_%A_%a.log
# F8/C8 retraining on the GalSBI re-simulation (training family 0-199). Experiment root, rule and settings:
# /project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_f8c8_20261102 (rule.md). Stages, in order:
#   sbatch -p cluster -c 1 --mem 8G -t 1:00:00 --array=0-19 jobs/job_galsbi_f8c8.sh domain     (10 cases per task)
#   sbatch -p cluster -c 1 --mem 8G -t 0:30:00 jobs/job_galsbi_f8c8.sh finalize
#   then the frozen F8 stages base (GPU), labels (CPU), refine (GPU) and the C8 stages; see rule.md.
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
R=/project/ls-gruen/users/zekang.zhang/sbsi_caches/galsbi_f8c8_20261102
SIMS=/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/training/sims
SBSI=/home/z/Zekang.Zhang/SBSI/.claude/worktrees/sampler-recipe
FROZEN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_f8_20260924_v1/venv/bin/python
C8PY=/project/ls-gruen/users/zekang.zhang/sbsi_caches/v36_classifier_f8_inputs_20260925_v1/venv/bin/python
active() {
  cd "$SBSI"
  /project/ls-gruen/users/zekang.zhang/envs/py31/bin/python -u -c 'import pathlib,runpy,sys;sys.path[:0]=[".","scripts"];import sbsi;assert pathlib.Path(sbsi.__file__).resolve().parent.parent==pathlib.Path.cwd();script=sys.argv.pop(1);runpy.run_path(script,run_name="__main__")' "$@"
}
hostname
case "$1" in
  domain)
    C0=$((SLURM_ARRAY_TASK_ID * 10))
    active scripts/prepare_galsbi_f8c8_domain.py --sims "$SIMS" --out "$R/harmonized/domain" --cases $(seq "$C0" $((C0 + 9))) ;;
  finalize)
    active scripts/prepare_galsbi_f8c8_domain.py --sims "$SIMS" --out "$R/harmonized/domain" --cases $(seq 0 199) \
      --finalize --c8-out "$R/c8/prepared" ;;
  base|labels|refine)
    cd /tmp && $FROZEN -I -u "$SBSI/scripts/run_galsbi_f8_stage.py" --root "$R" --stage "$1" ;;
  c8|c8smoke)
    # Frozen C8 trainer, unchanged; seed = SEED (20260913 primary, 20260914 replicate).
    : "${SEED:?}"
    cd /project/ls-gruen/users/zekang.zhang/sbsi_classifier_f8_inputs_20260925
    if [ "$1" = c8smoke ]; then
      $C8PY -u -m scripts.train_smooth_classifier --prepared-root "$R/c8/smoke_prepared" --output-root "$R/c8/smoke_C8_$SEED" \
        --arm smooth --feature-arm C8 --seed "$SEED" --device cpu --frozen-protocol "$R/c8/protocol.json" --smoke
    else
      $C8PY -u -m scripts.train_smooth_classifier --prepared-root "$R/c8/prepared" --output-root "$R/c8/C8_$SEED" \
        --arm smooth --feature-arm C8 --seed "$SEED" --device cuda --frozen-protocol "$R/c8/protocol.json"
    fi ;;
  *) exit 2 ;;
esac
