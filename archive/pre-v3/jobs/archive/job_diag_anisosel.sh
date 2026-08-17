#!/bin/bash
#SBATCH --job-name=anisosel
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/anisosel_%j.out

# STEP-0 GO/NO-GO for the joint measured+detection flow (#35): measure the SHAPE-CORRELATED
# (anisotropic) detection selection dS/dg = d<e1_input_rot0_p>_detected/dg on the two det_meas legs
# (g=0.0 train, g=0.05 val). e1_input_rot0_p is INTRINSIC (pre-shear) so any nonzero dS/dg is PURE
# selection. Needed ~0.02 in faint/blend cells to explain the oracle residual. DIAGNOSTIC ONLY;
# constgold never read; certified m untouched. Firewall-safe.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
echo "### ANISOSEL job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
stdbuf -oL -eL python -B -u scripts/diag_anisotropic_selection.py \
  || { echo "ANISOSEL FAILED"; exit 1; }
echo "### ANISOSEL_DONE job=$SLURM_JOB_ID ###"; date
