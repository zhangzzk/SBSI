#!/bin/bash
#SBATCH --job-name=lrsweep
#SBATCH --time=01:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/lrsweep_lr%a_%j.out
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
LAM_R=${LAM_R:?set LAM_R}
TAG=rblend_lr${LAM_R}_seed421
CK=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto/forward_${TAG}_joint.pt
echo "### LRSWEEP LAM_R=$LAM_R job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
# --- train flow-only with this lam_r (r_blend blend axis, lam_theta=0) ---
python -B -u scripts/train_joint_forward.py \
  --flow-catalogue $D/det_meas_crowd_g0.0_train_full.feather \
  --target-npz results/response_target_crowd_rblend_snc_c0-99_6x9x5.npz --decorrelate \
  --tag $TAG --flow-only \
  --max-case 200 --train-case-max 160 --det-max-case 100 --det-train-case-max 80 \
  --max-rows-flow 2400000 --max-rows-det 1600000 --max-rows-val 500000 --delta 0.05 \
  --lam-r $LAM_R --lam-theta 0 \
  --epochs 60 --patience 14 --no-sensitivity --no-baselines \
  --batch-size 16384 --lr 7e-4 --seed 421 \
  --context-dim 128 --set-hidden-dim 128 \
  --flow-equal-weight --flow-hidden-dim 256 --n-flows 10 --mean-hidden 128
echo "### train done, running half-shear eval ###"; date
python -B -u scripts/eval_halfshear_flowfig.py --max-case 19 --gtag g0.02 --ckpt $CK \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/halfshear_flowfig_lr${LAM_R}.npz
echo "### LRSWEEP_DONE LAM_R=$LAM_R job=$SLURM_JOB_ID ###"; date
