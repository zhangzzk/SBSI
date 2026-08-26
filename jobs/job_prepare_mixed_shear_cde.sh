#!/usr/bin/env bash
# Prepare the 200-case g=0.05 flow leg and its train-only trace+matrix target.

#SBATCH --job-name=sbsi_mixprep
#SBATCH --partition=cluster
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi_caches/mixed_shear_cde/logs/%x_%j.err

set -euo pipefail

repo=/home/z/Zekang.Zhang/SBSI
python=/project/ls-gruen/users/zekang.zhang/envs/sims1/bin/python
catalogues=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
cache=/project/ls-gruen/users/zekang.zhang/sbsi_caches
root=$cache/mixed_shear_cde

g0=$catalogues/det_meas_crowd_conc_g0.0_train_full.feather
g05_raw=$catalogues/det_meas_ngmix_g0.05_val.feather
g05=$root/det_meas_crowd_conc_g0.05_all200.feather
crowd=$cache/crowd_flux_conc_c0-199.feather
rblend=$catalogues/det_meas_crowd_g0.05_val_full.feather.oldcats_bak
target=$root/response_target_mixed_train160_6x6x5_matrix.npz

export PYTHONPATH=$repo
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p "$root/logs"

for path in "$g0" "$g05_raw" "$crowd" "$rblend"; do
  [[ -f "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
[[ ! -e "$target" ]] || { echo "refusing to overwrite $target" >&2; exit 2; }

if [[ -e "$g05" ]]; then
  echo "Reusing completed compact catalogue: $g05"
else
  echo "Preparing current-centroid g=0.05 compact catalogue for cases 0--199"
  date
  "$python" -u "$repo/scripts/prepare_flow_catalogue.py" \
    --input "$g05_raw" --output "$g05" \
    --crowd-lookup "$crowd" --rblend-lookup "$rblend" --max-case 200
fi

echo "Building response target on the deterministic 160 training cases only"
date
"$python" -u "$repo/scripts/build_flow_response_target.py" \
  --g0-catalogue "$g0" --sheared-catalogue "$g05" --output "$target" \
  --cases 200 --split-seed 501 --validation-size 0.2 \
  --n-flux 6 --n-size 6 --n-crowd 5 --min-count 500 \
  --primary-mag-max 25.8 --primary-re-min 0.5
date
echo "MIXED_SHEAR_PREP_DONE $g05 $target"
