#!/bin/bash
#SBATCH --job-name=joint_fwd99
#SBATCH --time=04:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/joint_fwd99_%j.out

# FIRST joint measurement+detection forward run of the SBSI reframe (GOALS.md, 2026-07-22).
# scripts/train_joint_forward.py vs the diagnostic prototype:
#   1. 4D measurement targets (measured shape + measured mag + measured size) -> mag/size are
#      OUTPUTS, killing the errors-in-variables floor. R_theta read out at eval, not pinned.
#   2. PRIMARY-ONLY shear in the response contexts -> the flow learns R_self ALONE (no R_blend).
#   3. TRUE-property primary cut Re_input_p>0.3 & true mag r_input_p<26 (defaults; neighbours full-pop).
# FIREWALL: flow = g=0 half-shear leg (det_meas_ngmix_g0.0_train, NON-ap7 to match the R_sim npz);
# det = g=0.05 half-shear leg (det_meas_g0.05_val, det_frac~0.44); shape R_sim = firewall-clean snc
# self-response LOOKUP; detection b_true = btrue_detection.npz. constgold is NEVER read here.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI

OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto
FLOWCAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_g0.0_train.feather
TARGET=results/response_target_isoblend_snc_c0-99_6x3x5.npz
SEED=${SEED:-421}
echo "### JOINT_FWD99 job=$SLURM_JOB_ID node=$SLURMD_NODENAME seed=$SEED ###"; nvidia-smi -L; date
echo "flow=$FLOWCAT"; echo "target-npz=$TARGET  (4D targets, primary-only shear, true-cut Re>0.3 & mag<26)"

stdbuf -oL -eL python -B -u scripts/train_joint_forward.py \
  --flow-catalogue "$FLOWCAT" \
  --target-npz "$TARGET" --decorrelate \
  --outdir "$OUTDIR" --tag "joint_c0-99_truecut_seed${SEED}" \
  --max-case 100 --train-case-max 80 \
  --max-rows-flow 1200000 --max-rows-det 1200000 --max-rows-val 500000 \
  --delta 0.05 \
  --lam-r 50 --lam-s 50 --lam-d 1 \
  --epochs 50 --patience 12 --no-sensitivity --no-baselines \
  --batch-size 16384 --lr 7e-4 --seed "$SEED" \
  --context-dim 128 --set-hidden-dim 128 --flow-hidden-dim 128 \
  --flow-layers 3 --n-flows 6 --mean-hidden 64 --det-hidden 128 \
  || { echo "JOINT_FWD99 FAILED seed=$SEED"; exit 1; }
echo "### JOINT_FWD99_DONE job=$SLURM_JOB_ID seed=$SEED ###"; date
