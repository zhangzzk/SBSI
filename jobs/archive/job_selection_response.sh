#!/bin/bash
#SBATCH --job-name SBSI_SELRESP
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_selresp_%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_selresp_%j.err

echo "START - SBSI selection-response gating diagnostic (classifier vs sim selection response)"
date
eval "$(conda shell.bash hook)"
conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"

CATDIR=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
C005="$CATDIR/det_meas_g0.05_val.feather"
C020="$CATDIR/det_meas_g0.2_val.feather"
MAXROWS="${MAXROWS:-800000}"

python -u SBSI/scripts/selection_response_diagnostic.py \
    --selection-model "SBSI/models/${SELMODEL:-selection_mlp_g0_shearfree_v1}.pt" \
    --catalogue-005 "$C005" --catalogue-020 "$C020" \
    --max-rows "$MAXROWS" \
    --output-dir SBSI/results/response_ratio \
    || echo "  (failed)"

echo "FINISH"; date
