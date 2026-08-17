#!/bin/bash
#SBATCH --job-name SBSI_PCAT_CLO
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=24G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_pcat_closure_%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_pcat_closure_%j.err

echo "START - meanblind p_cat (selection term) + closure tests"
date
eval "$(conda shell.bash hook)"
conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"

MODEL=SBSI/models/measurement_flow_g0_shape2d_meanblind_v1.pt
SEL=SBSI/models/selection_mlp_g0_shearfree_v1.pt
CATDIR=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
OUTDIR=SBSI/results/heldout_shear_recovery/measurement_flow_g0_shape2d_meanblind_v1
mkdir -p "$OUTDIR"

rec() { python -u SBSI/scripts/validate_heldout_shear_recovery.py \
          --measurement-model "$MODEL" --output-dir "$OUTDIR" \
          --which sheared "$@" || echo "  (failed: $*)"; }

echo "########## p_cat = p_meas * P(s=1)  (selection response included) ##########"
rec --catalogue "$CATDIR/det_meas_g0.05_val.feather" --nominal-shear 0.05 \
    --selection-model "$SEL" --max-rows 1000000
rec --catalogue "$CATDIR/det_meas_g0.2_val.feather"  --nominal-shear 0.2 \
    --selection-model "$SEL" --max-rows 1000000

echo "########## closure: targets drawn from the flow at known s0 (expect peak=s0) ##########"
rec --catalogue "$CATDIR/det_meas_g0.05_val.feather" --nominal-shear 0.05 \
    --closure-shear 0.05 --max-rows 300000
rec --catalogue "$CATDIR/det_meas_g0.2_val.feather"  --nominal-shear 0.2 \
    --closure-shear 0.2  --max-rows 300000

echo "FINISH"; date
