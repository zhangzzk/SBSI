#!/bin/bash
#SBATCH --job-name=s2_closure
#SBATCH --time=03:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/s2_closure_s%a_%j.out

# Constgold closure m for the "V1 + true-conditioning + 4D output" model (ablation rung S2).
# validate_constant_with_blend.py reads the checkpoint's declared condition_features
# (all TRUE props) and pulls them from constgold natively -> no meas-prim-lookup needed.
# m = R_sim/(R_flow + R_blend) - 1, directly comparable to certified V1 m=+0.245%.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation

SEED=$((501 + SLURM_ARRAY_TASK_ID))
CKDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CK=$CKDIR/measurement_flow_g0_ngmix_ablate_s2_true4d_s${SEED}_swaavg.pt
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
RES=/home/z/Zekang.Zhang/SBSI/results

echo "### S2 CONSTGOLD CLOSURE job=$SLURM_JOB_ID seed=$SEED ck=$CK ###"
nvidia-smi -L; date
[ -f "$CK" ]  || { echo "MISSING CK $CK"; exit 1; }
[ -f "$CAT" ] || { echo "MISSING CAT $CAT"; exit 1; }

python -u scripts/validate_constant_with_blend.py \
  --measurement-model "$CK" \
  --catalogue "$CAT" \
  --min-case 40 \
  --blend-lookup "$RES/blend_lookup_extnbrho_c40-139.feather" \
  --crowd-flux-lookup "$RES/crowd_flux_conc_c0-199.feather" \
  --global-only --flow-seed 12345 --n-samples 64 --max-rows 45000000 --batch-size 16384 \
  || { echo "S2_CLOSURE FAILED seed=$SEED"; exit 1; }
echo "S2_CLOSURE_DONE seed=$SEED"; date
