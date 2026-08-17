#!/bin/bash
#SBATCH --job-name=blmult4079
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/blmult4079_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
# c2-fix R_blend rebuild step 4c: rebuild blend_multiplicity_extnbrho_c40-79 on the RETRAINED emulator.
# The harvest (job_pilot_harvest.sh) reads blend_multiplicity_extnbrho_c40-79. build_blend_multiplicity
# runs the emulator on fix-invariant input truth, so it must be regenerated after the emulator retrain.
date
python -u scripts/build_blend_multiplicity.py --cases $(seq 40 79) \
  --tag lsst_r_extnbr_ho --output results/blend_multiplicity_extnbrho_c40-79.feather 2>&1 | grep -v "module command"
if [ -f results/blend_multiplicity_extnbrho_c40-79.feather ]; then date; echo BLMULT4079_DONE; else echo BLMULT4079_FAILED; exit 1; fi
