#!/bin/bash
#SBATCH --job-name SBSI_PCAT
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_pcat.%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_pcat.%j.err
echo "START - p_cat closure: does fixed classifier stop worsening m?"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /home/z/Zekang.Zhang/SBSI
MEAS=models/measurement_flow_g0_shape2d_respblend_lam1000_v1.pt
G02=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues/det_meas_g0.02_val.feather
echo "##### [A] NO selection (p_meas only) #####"
python -u scripts/validate_heldout_shear_recovery.py --measurement-model "$MEAS" \
    --catalogue "$G02" --nominal-shear 0.02 --which sheared \
    --grid-min -0.02 --grid-max 0.08 --grid-n 26 --max-rows 300000
echo "##### [B] OLD classifier (p_cat) -- expect WORSE #####"
python -u scripts/validate_heldout_shear_recovery.py --measurement-model "$MEAS" \
    --selection-model models/selection_mlp_g0_shearfree_v1.pt \
    --catalogue "$G02" --nominal-shear 0.02 --which sheared \
    --grid-min -0.02 --grid-max 0.08 --grid-n 26 --max-rows 300000
echo "##### [C] NEW response-aware classifier (p_cat) -- expect NOT worse #####"
python -u scripts/validate_heldout_shear_recovery.py --measurement-model "$MEAS" \
    --selection-model models/selection_respaware_lam300_v1.pt \
    --catalogue "$G02" --nominal-shear 0.02 --which sheared \
    --grid-min -0.02 --grid-max 0.08 --grid-n 26 --max-rows 300000
echo; echo "FINISH"; date
