#!/bin/bash
#SBATCH --job-name SBSI_RESP_G0
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=160G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_resp_g0_%j.out
#SBATCH --partition=inter
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_resp_g0_%j.err

echo "START - SBSI g=0-calibrated forward responsivity estimator"
date

eval "$(conda shell.bash hook)"
conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"

# Full-statistics run (all record batches): cubic/distortion/reduced forward responsivities
# + the decisive per-object forward-model fidelity check.
echo "########## no cut ##########"
python -u SBSI/scripts/responsivity_estimator_g0.py \
    --g0-max-rows 4000000 --max-rows 4000000

echo "FINISH"
date
