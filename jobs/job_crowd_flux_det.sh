#!/bin/bash
#SBATCH --job-name=crowd_flux_det
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/crowd_flux_det_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# Detected-only crowd-flux lookups for the etilde probabilistic-blending ladder
# (WORKLOG cont.32): step 1 = detected neighbours at TRUE flux (undetected-census
# effect); step 2 = detected neighbours at MEASURED flux_auto (deployment fluxes).
# Consumed by scripts/infer_posterior_shape.py via --crowd-flux-lookup (no code change).
CASES=$(seq 40 139)
# outputs are GB-scale -> $DATA_DIR (a results/ write blew the home quota and killed
# job 15062408); symlink into results/ afterwards if convenient
OUT=${OUT:-/project/ls-gruen/users/zekang.zhang/sbsi_caches}

echo "### detected-only, TRUE flux ###"; date
stdbuf -oL -eL python -u scripts/build_crowding_lookup_det.py --cases $CASES \
  --flux true --output "$OUT/crowd_flux_det_c40-139.feather" \
  2>&1 | grep --line-buffered -vE "module command"
date
# NOTE --flux measured NOT run: det_meas_crowd only covers its sampled subset (~half of
# detected neighbours lack measured mags) -> rung 2 needs the per-case measurement
# catalogues as the flux source; deferred.
echo "### DONE ###"
