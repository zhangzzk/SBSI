#!/bin/bash
#SBATCH --job-name=ood_lk
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/ood_lk_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
python -u scripts/build_ood_lookup.py --cases $(seq 0 39) \
  --base /project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant --sign 0.02 \
  --output results/ood_lookup_const_c0-39.feather 2>&1 | grep -v module
echo OOD_LK_DONE
