#!/bin/bash
#SBATCH --job-name=det_resp_eval
#SBATCH --time=00:40:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/det_resp_eval_%j.out

# Evaluate both trained detection classifiers (BCE-only lam0, response-aware lam300): induced
# detection response b_model vs sim b_sim, global (un-binned observable ~R_detect) + per blend bin.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI-ablation
B=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk
echo "### DET RESP EVAL job=$SLURM_JOB_ID ###"; date
for LAM in 0 300; do
  echo "===== eval lam=$LAM ====="
  python -u scripts/eval_detection_response.py \
    --model $B/det_response_mlp_lam${LAM}_s7.pt \
    --response-target-npz $B/det_response_target_g05.npz \
    --max-rows 4000000 --seed 99 \
    --output $B/det_response_eval_lam${LAM}.npz
done
echo "DET_RESP_EVAL_JOB_DONE"; date
