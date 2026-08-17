#!/bin/bash
#SBATCH --job-name=joint_triad
#SBATCH --time=01:30:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/joint_triad_%j.out

# SIM-side response TRIAD on constgold (GOALS.md reframe): isolate SHAPE / SELECTION / DETECTION
# bias with the orthogonal design (owner spec 2026-07-22). EVALUATION ONLY — reads constgold (the
# held-out acceptance set); trains nothing. Full train catalogue for tight case-bootstrap errors.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"

OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/joint_triad_train.npz
echo "### JOINT_TRIAD job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -B -u scripts/eval_joint_triad.py \
  --true-re-min 0.3 --true-mag-max 26 \
  --sn-cuts 7 10 15 20 \
  --n-boot 300 \
  --output "$OUT" \
  || { echo "JOINT_TRIAD FAILED"; exit 1; }
echo "### JOINT_TRIAD_DONE job=$SLURM_JOB_ID ###"; date
