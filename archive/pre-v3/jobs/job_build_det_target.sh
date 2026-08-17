#!/bin/bash
#SBATCH --job-name=det_target
#SBATCH --time=00:40:00
#SBATCH --mem=110G
#SBATCH --cpus-per-task=8
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/det_target_%j.out

# Build the sim DETECTION-response target b_sim(true flux x size x blend) on the g=0.05 sheared
# parent (det+undet). This is the -~2% target the response-aware classifier is regularized toward.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI-ablation
echo "### DET TARGET job=$SLURM_JOB_ID ###"; date
python -u scripts/build_detection_response_target.py \
  --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.05_val.feather \
  --nominal-g 0.05 --n-flux 4 --n-size 2 --n-dist 3 --max-rows 20000000 \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/det_response_target_g05.npz
echo "DET_TARGET_JOB_DONE"; date
