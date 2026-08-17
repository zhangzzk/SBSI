#!/bin/bash
#SBATCH --job-name=ra_regress
#SBATCH --time=00:20:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/ra_regress_%j.out
set -o pipefail
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
R=/home/z/Zekang.Zhang/SBSI/results
D=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
S=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ra/smoke
echo "### FIDUCIAL REGRESSION (no RA flags at all) job=$SLURM_JOB_ID ###"; date
python -u scripts/train_measurement_model_swa_s1_truecond.py \
  --catalogue $D/det_meas_crowd_conc_g0.0_train_full.feather --output $S/flow_fid_regress.pt \
  --target-column detected --selection-name sextractor_detected --feature-set g0_meas_crowd_conc_szfl_noz \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type mean_affine --mean-hidden 128 --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 --max-rows 40000 --max-read-batches 8 --epochs 2 --batch-size 4096 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 5 --weight-decay 1e-5 \
  --seed 501 --num-workers 4 --gpu-resident --swa-last-k 2 \
  --primary-mag-max 26.0 --primary-re-min 0.3 \
  --response-weight 450 --response-delta 0.02 --response-difference central \
  --response-target-npz $R/response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz \
  --coupling-weight 500 --coupling-target-npz $R/response_target_theta_coupling_rblend_c0-99_6x9x5.npz \
  || { echo FID_REGRESS_FAILED; exit 1; }
echo; echo "=== forward-difference / no-coupling branch (batch length 7) ==="
python -u scripts/train_measurement_model_swa_s1_truecond.py \
  --catalogue $D/det_meas_crowd_conc_g0.0_train_full.feather --output $S/flow_fid_fwd.pt \
  --target-column detected --selection-name sextractor_detected --feature-set g0_meas_crowd_conc_szfl_noz \
  --target-features measured_ngmix_g1 measured_ngmix_g2 measured_mag_auto measured_log_flux_radius \
  --flow-type mean_affine --mean-hidden 128 --flow-blind-features e1_input_p e2_input_p \
  --shear-case 0.0 --max-rows 20000 --max-read-batches 4 --epochs 1 --batch-size 4096 \
  --hidden-dim 256 --condition-layers 3 --n-flows 10 --lr 0.0007 --patience 5 --weight-decay 1e-5 \
  --seed 501 --num-workers 4 --gpu-resident --swa-last-k 1 \
  --response-weight 450 --response-delta 0.02 --response-difference forward \
  --response-target-npz $R/response_target_crowd_rblend_snc_c0-99_6x6x5_dom.npz \
  || { echo FWD_REGRESS_FAILED; exit 1; }
echo FID_REGRESS_DONE; date
