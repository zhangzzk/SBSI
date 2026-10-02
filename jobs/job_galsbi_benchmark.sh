#!/bin/bash
#SBATCH --job-name=sbsi-mh-cost
#SBATCH --partition=cip
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:25:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
: "${SBSI_BENCH_ROOT:?set benchmark root}"
: "${SBSI_BENCH_DEVICE:?set cpu or cuda}"
: "${SBSI_CACHE_DIR:?set artifact root}"
: "${BLENDEMU_RUNS_DIR:?set response artifact root}"
SBSI_BENCH_PYTHON=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
SBSI_BENCH_OUTPUT="$SBSI_BENCH_ROOT/$SBSI_BENCH_DEVICE"
mkdir -p "$SBSI_BENCH_OUTPUT"
cd "${SBSI_BENCH_SNAPSHOT:-$SBSI_BENCH_ROOT/source_snapshot}"
sha256sum --check SHA256SUMS --quiet
"$SBSI_BENCH_PYTHON" /home/z/Zekang.Zhang/.agents/skills/get-available-resources/scripts/detect_resources.py --output "$SBSI_BENCH_OUTPUT/resources.json"
SBSI_BENCH_SIZES=(4 32 128 512 1024)
if [[ $SBSI_BENCH_DEVICE == cpu ]]; then
    SBSI_BENCH_SIZES=(4 32 128)
fi
"$SBSI_BENCH_PYTHON" -u -c 'import pathlib, runpy, sys; sys.path[:0]=[".", "scripts"]; import sbsi; assert pathlib.Path(sbsi.__file__).resolve().parent.parent == pathlib.Path.cwd(); runpy.run_path("scripts/benchmark_galsbi_sampler.py", run_name="__main__")' \
    --reference /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_summary_mh_20260930_v2 \
    --scenes /project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/galsbi_long_mh_20261001/seed2026100121/final_scenes.npz \
    --output "$SBSI_BENCH_OUTPUT" --device "$SBSI_BENCH_DEVICE" \
    --observations "${SBSI_BENCH_SIZES[@]}" --repeats 8
