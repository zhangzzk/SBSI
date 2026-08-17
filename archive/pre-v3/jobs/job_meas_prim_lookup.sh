#!/bin/bash
#SBATCH --job-name=measprimlk
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/measprimlk_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

# Build the constgold measured-primary lookup (WORKLOG cont.19) so realistic flows can be validated.
# Averages measured mag_auto/flux_radius/class_star over the +/-0.02 renders per (case,input_index).
OUT=results/meas_prim_lookup_c0-139.feather
echo "### build measured-primary lookup, cases 0-139 -> $OUT ###"; date
python -u scripts/build_meas_prim_lookup.py --cases $(seq 0 139) --output $OUT 2>&1 | grep -v "module command"
date; echo "MEAS_PRIM_LOOKUP_DONE $OUT"
