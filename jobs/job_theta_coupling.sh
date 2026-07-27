#!/bin/bash
#SBATCH --job-name=theta_coup
#SBATCH --time=00:40:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/theta_coup_%j.out

# Build the R_theta ORIENTATION-COUPLING target (owner: pin the coupling directly). Per flow-grid
# cell, slope b_size = d(log measured_flux_radius)/dg vs e_int.ghat (the size selection driver);
# b_mag ~ 0. Measured on the matched half-shear legs (g0 vs g05, neighbour fixed). Grid edges copied
# from the shape target so the per-object binid is identical. Firewall: half-shear only.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI

OUT=results/response_target_theta_coupling_c0-99_6x9x5.npz
echo "### THETA_COUP job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -B -u scripts/build_theta_coupling_target.py \
  --grid-npz results/response_target_isoblend_RAWfine_c0-99_6x9x5.npz \
  --max-case 99 --min-count 300 \
  --output "$OUT" \
  || { echo "THETA_COUP FAILED"; exit 1; }
echo "### THETA_COUP_DONE job=$SLURM_JOB_ID ###"; date
