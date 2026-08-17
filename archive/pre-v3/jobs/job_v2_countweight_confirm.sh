#!/bin/bash
#SBATCH --job-name=v2_countw
#SBATCH --time=02:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --array=0-2
#SBATCH --output=/home/z/Zekang.Zhang/logs/v2_countw_s%a_%j.out
set -e

# CONFIRMING RUN: V2 flow-only, TRUE-cond, 6x9x5 target, EXACTLY the ens_lr250_swa8 baseline
# (.claude/jobs/e1915e2b/tmp/job_ens_lr250.sh) EXCEPT --flow-equal-weight is DROPPED (count-weighted
# response loss, matching the V1 ladder's weighting) and --max-rows-flow raised to 4,000,000 (owner
# override, for a cleaner/faster capped run; eval response mean is robust to the train row cap).
# 3 seeds 501/502/503. Runs the MAIN checkout's train_joint_forward.py READ-ONLY (not modified);
# only this job lives in the worktree. Ckpts -> sbsi_caches/ablation/forward_<tag>_joint.pt.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
OUTDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
mkdir -p "$OUTDIR"
SEED=$((501 + SLURM_ARRAY_TASK_ID))
TAG=countw_lr250_swa8_seed${SEED}
echo "### V2_COUNTW seed=$SEED job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date

python -B -u scripts/train_joint_forward.py \
  --flow-catalogue $D/det_meas_crowd_g0.0_train_full.feather \
  --target-npz results/response_target_crowd_rblend_snc_c0-99_6x9x5.npz --decorrelate \
  --outdir "$OUTDIR" --tag $TAG --flow-only \
  --max-case 200 --train-case-max 160 --det-max-case 100 --det-train-case-max 80 \
  --max-rows-flow 4000000 --max-rows-det 1600000 --max-rows-val 500000 --delta 0.05 \
  --lam-r 250 --lam-theta 0 \
  --epochs 60 --patience 14 --swa-last-k 8 --no-sensitivity --no-baselines \
  --batch-size 16384 --lr 7e-4 --seed $SEED \
  --context-dim 128 --set-hidden-dim 128 \
  --flow-hidden-dim 256 --n-flows 10 --mean-hidden 128
echo "### V2_COUNTW_DONE seed=$SEED job=$SLURM_JOB_ID MODEL=$OUTDIR/forward_${TAG}_joint.pt ###"; date
