#!/bin/bash
#SBATCH --job-name=rich_gate
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=cluster
#SBATCH --output=/home/z/Zekang.Zhang/logs/rich_gate_%j.out

# DECISION GATE (Goal-1 size-robustness, task #32): build fine-size forward-diff response target
# grids at g=0.02 and g=0.05 (matched cases 0-19, SNC vs the g=0 lookup, DEFAULT non-circular mode),
# then test whether Richardson (5/3 R02 - 2/3 R05) cancels the forward-diff offset vs constgold truth
# at small size. EVALUATION/target-build only; constgold used only as the truth to compare against,
# never to build a target. No certified artifact touched.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
LK=results/g0_lookup_c0-99.feather
EDGES="0.1,0.2,0.3,0.5,0.75,1.0,1.5"
G02=results/rtgt_g02_fine_c0-19.npz
G05=results/rtgt_g05_fine_c0-19.npz
echo "### RICH_GATE job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date

echo "===== build g=0.02 forward-diff target grid (cases 0-19) ====="
stdbuf -oL -eL python -B -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_ngmix_g0.02_test.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.02 --n-flux 6 --size-edges "$EDGES" --n-dist 4 \
  --min-count 300 --max-case 19 \
  --snc-lookup "$LK" --snc-cols ngmix0_g1 ngmix0_g2 \
  --output $G02 2>&1 | grep -v module || { echo "BUILD_G02 FAILED"; exit 1; }

echo "===== build g=0.05 forward-diff target grid (cases 0-19, matched) ====="
stdbuf -oL -eL python -B -u scripts/compute_response_target_blend.py \
  --catalogue $D/det_meas_ngmix_g0.05_val.feather \
  --target-cols measured_ngmix_g1 measured_ngmix_g2 \
  --nominal-g 0.05 --n-flux 6 --size-edges "$EDGES" --n-dist 4 \
  --min-count 300 --max-case 19 \
  --snc-lookup "$LK" --snc-cols ngmix0_g1 ngmix0_g2 \
  --output $G05 2>&1 | grep -v module || { echo "BUILD_G05 FAILED"; exit 1; }

echo "===== GATE: Richardson vs constgold truth by size ====="
stdbuf -oL -eL python -B -u scripts/diag_richardson_gate.py \
  --g02 $G02 --g05 $G05 --max-case 19 || { echo "GATE_DIAG FAILED"; exit 1; }
echo "### RICH_GATE_DONE job=$SLURM_JOB_ID ###"; date
