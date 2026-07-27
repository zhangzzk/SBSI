#!/bin/bash
#SBATCH --job-name=est_match
#SBATCH --time=01:30:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/est_match_%j.out

# Cross-estimator responsivity match + firewall-clean R_blend (loop cont.114). The flow trains on
# det_meas_ngmix (ngmix shapes); the deliverable is on constgold (measured_e, a DIFFERENT estimator +
# sim set, ~10% higher responsivity). This measures, on the TRUE-CUT population:
#   * isolated R_self on ngmix (ngmix units) vs constgold (constgold units) -> the responsivity BRIDGE
#   * blended coherent R1 on constgold, minus ngmix R_self(blended) -> the R_blend the emulator must hit
# EVALUATION ONLY: reads the ngmix half-shear leg (train-safe) + constgold (held-out); trains nothing.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /home/z/Zekang.Zhang/SBSI

OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/estimator_match.npz
echo "### EST_MATCH job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; date
python -B -u scripts/eval_estimator_match.py \
  --true-re-min 0.3 --true-mag-max 26 \
  --ngmix-max-rows 40000000 \
  --cg-max-case 40 \
  --output "$OUT" \
  || { echo "EST_MATCH FAILED"; exit 1; }
echo "### EST_MATCH_DONE job=$SLURM_JOB_ID ###"; date
