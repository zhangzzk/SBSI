#!/bin/bash
#SBATCH --job-name=CG_lam300
#SBATCH --time=01:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/cg_lam300_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:${BLENDEMU_ROOT:-/home/z/Zekang.Zhang/blendemu}:$PYTHONPATH"; cd "${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}"
python -u scripts/validate_constant_response.py \
  --measurement-model models/measurement_flow_g0_ngmix_ap7_respblend_lam300_v1.pt \
  --max-rows 6000000 --batch-size 8192
echo CONST_GOLD_DONE
