#!/bin/bash
#SBATCH --job-name=NP7_train
#SBATCH --time=06:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/np7_train_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/np7_train_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
LAM=${1:-300}
python -u scripts/train_measurement_model.py --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_np7_g0.0_train.feather   --output models/measurement_flow_g0_ngmix_np7_respblend_lam${LAM}_v1.pt   --target-column detected --selection-name sextractor_detected --feature-set g0_oriented   --target-features measured_ngmix_g1 measured_ngmix_g2 --flow-type mean_affine --mean-hidden 128   --flow-blind-features e1_input_p e2_input_p --shear-case 0.0 --max-rows 10000000   --epochs 100 --batch-size 8192 --hidden-dim 256 --condition-layers 3 --n-flows 10   --lr 0.0007 --patience 12 --num-workers 8 --response-weight "$LAM" --response-delta 0.05   --response-target-npz results/response_target_np7_6x3x4.npz || { echo TRAIN FAILED; exit 1; }
echo "NP7 TRAIN DONE"
