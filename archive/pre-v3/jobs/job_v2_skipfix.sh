#!/bin/bash
#SBATCH --job-name=v2_skipfix
#SBATCH --time=02:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --array=0-2
#SBATCH --output=/home/z/Zekang.Zhang/logs/v2_skipfix_s%a_%j.out
set -e

# ARCHITECTURE-FIX EXPERIMENT: the V2-count baseline PLUS a DIRECT intrinsic-shape skip into the
# mean head (--shape-skip): e1/e2_input_p feed the mean head directly, bypassing the DeepSets trunk
# that (hypothesis) smears the shape->response mapping and causes V2's ISO under-response.
# Runs the WORKTREE-EDITED train_joint_forward.py + sbs_shear/forward_model.py (worktree FIRST on
# PYTHONPATH so the edits win; the MAIN checkout is untouched). Everything else = V2-count baseline
# (flow-only, true-cond, 6x9x5 target, COUNT-weighted, --max-rows-flow 4M). 3 seeds 501-503.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
RESP=/home/z/Zekang.Zhang/SBSI/results/response_target_crowd_rblend_snc_c0-99_6x9x5.npz
OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
mkdir -p "$OUTDIR"
SEED=$((501 + SLURM_ARRAY_TASK_ID))
TAG=skipfix_lr250_swa8_seed${SEED}
echo "### V2_SKIPFIX seed=$SEED job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
[ -f "$RESP" ] || { echo "MISSING RESP $RESP"; exit 1; }

python -B -u scripts/train_joint_forward.py \
  --flow-catalogue $D/det_meas_crowd_g0.0_train_full.feather \
  --target-npz "$RESP" --decorrelate \
  --outdir "$OUTDIR" --tag $TAG --flow-only \
  --shape-skip \
  --max-case 200 --train-case-max 160 --det-max-case 100 --det-train-case-max 80 \
  --max-rows-flow 4000000 --max-rows-det 1600000 --max-rows-val 500000 --delta 0.05 \
  --lam-r 250 --lam-theta 0 \
  --epochs 60 --patience 14 --swa-last-k 8 --no-sensitivity --no-baselines \
  --batch-size 16384 --lr 7e-4 --seed $SEED \
  --context-dim 128 --set-hidden-dim 128 \
  --flow-hidden-dim 256 --n-flows 10 --mean-hidden 128
echo "### V2_SKIPFIX_DONE seed=$SEED job=$SLURM_JOB_ID MODEL=$OUTDIR/forward_${TAG}_joint.pt ###"; date
