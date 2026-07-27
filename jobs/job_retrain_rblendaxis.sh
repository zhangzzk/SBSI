#!/bin/bash
#SBATCH --job-name=rblendaxis
#SBATCH --time=02:30:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/rblendaxis_%j.out
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
EPOCHS=${EPOCHS:-60}; MRF=${MRF:-2400000}; MRV=${MRV:-500000}; TAG=${TAG:-rblendaxis_seed421}
echo "### RBLENDAXIS job=$SLURM_JOB_ID node=$SLURMD_NODENAME TAG=$TAG EPOCHS=$EPOCHS ###"; nvidia-smi -L; date
python -c "import scripts.train_joint_forward as m; print('trainer module:', m.__file__)"
python -B -u scripts/train_joint_forward.py \
  --flow-catalogue $D/det_meas_crowd_g0.0_train_full.feather \
  --target-npz results/response_target_crowd_rblend_snc_c0-99_6x9x5.npz --decorrelate \
  --tag $TAG \
  --max-case 200 --train-case-max 160 --det-max-case 100 --det-train-case-max 80 \
  --max-rows-flow $MRF --max-rows-det 1600000 --max-rows-val $MRV --delta 0.05 \
  --lam-r 450 --lam-s 50 --lam-d 1 --lam-theta 0 \
  --epochs $EPOCHS --patience 14 --no-sensitivity --no-baselines \
  --batch-size 16384 --lr 7e-4 --seed 421 \
  --context-dim 128 --set-hidden-dim 128 \
  --flow-equal-weight --flow-hidden-dim 256 --n-flows 10 --mean-hidden 128
echo "### RBLENDAXIS_DONE job=$SLURM_JOB_ID TAG=$TAG ###"; date
