#!/bin/bash
# Train decorrelated (shape<->size independent) flow variants, then chain a 1M-row baseline
# recovery for each. Tests whether removing the isolated covariate-shift source at the TRAINING
# level reduces m. Two variants vs their non-decorrelated controls:
#   decorr_meanblind  vs meanblind (+0.030)  -- can we beat the current best?
#   decorr_meanmlp    vs meanmlp   (+0.068)  -- clean test on a faithful (non-underfit) model
# Submit:  VARIANT=meanblind sbatch job_decorr_sweep.sh ;  VARIANT=meanmlp sbatch job_decorr_sweep.sh
#SBATCH --job-name SBSI_DECORR
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=zekang.zhang@physik.lmu.de
#SBATCH --chdir=/home/z/Zekang.Zhang
#SBATCH --output=/home/z/Zekang.Zhang/logs/sbsi_decorr_%j.out
#SBATCH --partition=cip
#SBATCH -e /home/z/Zekang.Zhang/logs/sbsi_decorr_%j.err

echo "START decorr sweep variant=${VARIANT:?set VARIANT=meanblind|meanmlp}"; date
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="${SBSI_ROOT:-/home/z/Zekang.Zhang/SBSI}:$PYTHONPATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
CATDIR=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues

case "$VARIANT" in
  meanblind) MH=0  ;;   # linear mean head (matches the +0.030 best)
  meanmlp)   MH=64 ;;   # MLP mean head (faithful control, was +0.068)
  *) echo "unknown VARIANT"; exit 1 ;;
esac
TAG="measurement_flow_g0_shape2d_decorr_${VARIANT}_v1"
MODEL="SBSI/models/${TAG}.pt"

echo "===== TRAIN $TAG (mean-hidden=$MH, blind shape, DECORRELATED) ====="
python -u SBSI/scripts/train_measurement_model.py \
    --catalogue "$CATDIR/det_meas_g0.0_train.feather" \
    --output "$MODEL" \
    --target-column detected --selection-name sextractor_detected \
    --feature-set g0_oriented \
    --target-features measured_e1_image measured_e2_image \
    --flow-type mean_affine --mean-hidden "$MH" \
    --flow-blind-features e1_input_p e2_input_p \
    --decorrelate-shape-size \
    --shear-case 0.0 --max-rows 6000000 \
    --epochs 150 --batch-size 8192 --hidden-dim 256 --condition-layers 3 \
    --n-flows 10 --lr 0.0007 --patience 15 --num-workers 8 || { echo "TRAIN FAILED"; exit 1; }

OUTDIR="SBSI/results/heldout_shear_recovery/${TAG}"
mkdir -p "$OUTDIR"
echo "===== RECOVER $TAG (1M rows, baseline) ====="
for sp in "0.05:det_meas_g0.05_val.feather" "0.2:det_meas_g0.2_val.feather"; do
  nom="${sp%%:*}"; cat="${sp##*:}"
  python -u SBSI/scripts/validate_heldout_shear_recovery.py \
      --measurement-model "$MODEL" --catalogue "$CATDIR/$cat" \
      --nominal-shear "$nom" --which sheared \
      --output-dir "$OUTDIR" --max-rows 1000000 || echo "  (recover failed: $nom)"
done

echo "FINISH"; date
