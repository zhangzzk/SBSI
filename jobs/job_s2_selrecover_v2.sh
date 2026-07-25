#!/bin/bash
#SBATCH --job-name=s2_selrec2
#SBATCH --time=02:00:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/s2_selrec2_%j.out

# Selection recovery on flux/size, cuts bracketing the training limits (mag 24.5..26.5;
# size 0.2..0.8" = 1.0..4.0 px, covering the requested 0.2-0.4" plus the real selecting edge ~0.55-0.8").
# Saves per-cut R_sim/R_sim_err + per-seed R_model spread -> npz for the dots+error-bar plot.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CAT=/project/ls-gruen/users/zekang.zhang/sbsi_catalogues
OUT=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/selrecover_v2_lt500.npz
echo "### S2 SELREC v2 job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_selection_response.py \
  --ckpt-glob "$D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt500_s50*_swaavg.pt" \
  --g0-leg $CAT/det_meas_ngmix_g0.0_train.feather \
  --gS-leg $CAT/det_meas_ngmix_g0.02_test.feather \
  --max-case 39 --n-samples 128 --batch-size 16384 \
  --size-cuts 1.0 1.5 2.0 2.5 3.0 3.5 4.0 \
  --mag-cuts 24.5 25.0 25.5 26.0 26.5 \
  --true-size-cuts 0.4 0.5 --output $OUT
echo "S2_SELREC2_DONE"; date
