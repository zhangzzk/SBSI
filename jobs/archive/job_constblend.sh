#!/bin/bash
#SBATCH --job-name=CG_blend
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_blend_%j.out
#SBATCH --error=/home/z/Zekang.Zhang/logs/cg_blend_%j.err
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
cd /home/z/Zekang.Zhang/SBSI
python -u scripts/validate_constant_with_blend.py \
  --measurement-model models/measurement_flow_g0_ngmix_ap7_respblend_lam300_v1.pt \
  --max-rows 6000000 --n-samples 128
echo CG_BLEND_DONE
