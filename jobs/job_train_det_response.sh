#!/bin/bash
#SBATCH --job-name=det_resp
#SBATCH --time=01:10:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/det_resp_%j.out

# Train the detection classifier P(detect | true props + blend), shear-free (g0) features, with the
# shear-response regularizer BCE + LAM*(b_model-b_sim)^2 per flux/size/blend bin. Pass LAM/OUT/SEED
# via --export. LAM=0 -> BCE-only baseline (expected wrong-sign response ~+1.9%); LAM=300 -> corrected.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation
LAM=${LAM:-300}
SEED=${SEED:-7}
OUT=${OUT:-/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/det_response_mlp_lam${LAM}_s${SEED}.pt}
echo "### DET RESP job=$SLURM_JOB_ID  LAM=$LAM SEED=$SEED OUT=$OUT ###"; date
python -u scripts/train_detection_response.py \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.0_train.feather \
  --response-target-npz /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/det_response_target_g05.npz \
  --feature-set g0_shearfree \
  --response-weight $LAM --response-delta 0.05 \
  --max-rows 8000000 --epochs 60 --batch-size 32768 --lr 1e-3 --seed $SEED \
  --output $OUT
echo "DET_RESP_JOB_DONE"; date
