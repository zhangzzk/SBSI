#!/usr/bin/env bash
# Render the already-computed 20-case complete-forward property comparison.

#SBATCH --job-name=sbsi_cg_complete20_plot
#SBATCH --partition=inter
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_complete_c40_59_v1/logs/%x_%j.out
#SBATCH --error=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_complete_c40_59_v1/logs/%x_%j.err

set -euo pipefail

repo=${SBSI_REPOSITORY:-/home/z/Zekang.Zhang/SBSI}
python=${SBSI_ANALYSIS_PYTHON:-/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python}
root=/project/ls-gruen/users/zekang.zhang/sbsi/catalogue_prior/inference_constgold_plus_size075_n100k_m16384_v1/forward_response_complete_c40_59_v1
samples=$root/contour_samples
figures=$root/property_contours

[[ -s "$root/result.json" ]] || { echo "missing response result" >&2; exit 2; }
[[ ! -e "$figures" ]] || { echo "refusing to overwrite $figures" >&2; exit 2; }
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"
exec "$python" -u -m scripts.plot_complete_forward_properties \
  --predicted "$samples/predicted_plus_selected_100k.parquet" \
  --measured "$samples/measured_plus_selected_100k.parquet" \
  --sample-manifest "$samples/manifest.json" \
  --output "$figures" --pixel-size 0.2
