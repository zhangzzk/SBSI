#!/bin/bash
#SBATCH --job-name=s2c_cg
#SBATCH --time=03:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/s2c_cg_t%a_%j.out

# Constgold closure m for the coupling-pinned models: shape MUST stay ~+0.2% (= S2) while size is fixed.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation
LAMS=(100 500)
LT=${LAMS[$SLURM_ARRAY_TASK_ID]}
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CK=$D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt${LT}_s501_swaavg.pt
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
RES=/home/z/Zekang.Zhang/SBSI/results
echo "### S2C CONSTGOLD lam_theta=$LT ck=$CK ###"; nvidia-smi -L; date
[ -f "$CK" ] || { echo "MISSING CK"; exit 1; }
python -u scripts/validate_constant_with_blend.py \
  --measurement-model "$CK" --catalogue "$CAT" --min-case 40 \
  --blend-lookup "$RES/blend_lookup_extnbrho_c40-139.feather" \
  --crowd-flux-lookup "$RES/crowd_flux_conc_c0-199.feather" \
  --global-only --flow-seed 12345 --n-samples 64 --max-rows 45000000 --batch-size 16384 \
  || { echo "S2C_CG FAILED lam=$LT"; exit 1; }
echo "S2C_CG_DONE lam=$LT"; date
