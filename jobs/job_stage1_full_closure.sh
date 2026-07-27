#!/bin/bash
#SBATCH --job-name=s1_full
#SBATCH --time=01:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/s1_full_%j.out
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI
CKPT="/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto/forward_sw_th400_seed*_joint.pt"
NN=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/nn_dist_c0-39.feather
echo "### S1_FULL job=$SLURM_JOB_ID node=$SLURMD_NODENAME ###"; nvidia-smi -L; date
for RB in d7 ""; do
  if [ "$RB" = "d7" ]; then BLEND=results/blend_lookup_extnbrho_d7_c0-39.feather; TAG="7arcsec-gated";
  else BLEND=results/blend_lookup_extnbrho_c0-39.feather; TAG="10arcsec-native"; fi
  echo; echo "########## sw_th400 + R_blend=$TAG ($BLEND) + 7\" isolation ##########"
  python -B -u scripts/eval_constgold_closure.py --ckpt-glob "$CKPT" --bridge 1.0 \
    --nn-lookup "$NN" --iso-radius 7.0 --blend-lookup "$BLEND" \
    --max-case 39 --true-re-min 0.3 --true-mag-max 26 \
    --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/stage1_full_${TAG}.npz
done
echo "### S1_FULL_DONE job=$SLURM_JOB_ID ###"; date
