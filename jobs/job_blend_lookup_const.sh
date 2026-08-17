#!/bin/bash
#SBATCH --job-name=blend_lu
#SBATCH --time=03:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/blend_lu_%j.out
# Firewall-clean summed-aperture R_blend on constgold via the BlendEMU emulator (cont.116).
# Supplies the +0.10 the blended band needs. Emulator trained on half-shear; applied to constgold INPUTS.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
OUT=results/blend_lookup_const_c0-40.feather
echo "### BLEND_LU job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -B -u scripts/build_blend_lookup.py --cases $(seq 0 40) --output "$OUT" \
  || { echo "BLEND_LU FAILED"; exit 1; }
echo "### BLEND_LU_DONE job=$SLURM_JOB_ID ###"; date
