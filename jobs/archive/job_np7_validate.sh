#!/bin/bash
#SBATCH --job-name=NP7_val
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=cip
#SBATCH --output=/home/z/Zekang.Zhang/logs/np7_val_%j.out
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"; cd /home/z/Zekang.Zhang/SBSI
M=models/measurement_flow_g0_ngmix_np7_respblend_lam300_v1.pt; LK=results/g0_lookup_c0-99.feather
echo "### NP7 g=0.05 (SNC) ###"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_np7_g0.05_val.feather --nominal-g 0.05 --max-rows 0 --n-samples 64 --snc-lookup "$LK" 2>&1 | grep -iE "SNC ON|GLOBAL|ISOLATED|nbdist" | grep -v module
echo "### NP7 g=0.02 (SNC, independent) ###"
python -u scripts/validate_allpairs_response.py --measurement-model "$M" --catalogue /project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_ngmix_np7_g0.02_test.feather --nominal-g 0.02 --max-rows 0 --n-samples 64 --snc-lookup "$LK" 2>&1 | grep -iE "SNC ON|GLOBAL|ISOLATED|nbdist" | grep -v module
echo NP7_VALIDATE_DONE
