#!/bin/bash
#SBATCH --job-name=s2c_final
#SBATCH --time=03:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/s2c_final_t%a_%j.out

eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
RES=/home/z/Zekang.Zhang/SBSI/results
echo "### S2C FINAL task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID ###"; nvidia-smi -L; date

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  # 3-seed flux/size response ensemble (lt=500) + ghat-leg shape control
  python -u scripts/eval_fluxsize_response.py \
    --ckpt-glob "$D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt500_s50*_swaavg.pt" \
    --max-case 39 --output "$D/fluxsize_resp_s2c_lt500_3seed.npz"
else
  # constgold m for seeds 502 / 503
  SEEDS=(_ 502 503)
  S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
  CK=$D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt500_s${S}_swaavg.pt
  python -u scripts/validate_constant_with_blend.py \
    --measurement-model "$CK" --catalogue "$CAT" --min-case 40 \
    --blend-lookup "$RES/blend_lookup_extnbrho_c40-139.feather" \
    --crowd-flux-lookup "$RES/crowd_flux_conc_c0-199.feather" \
    --global-only --flow-seed 12345 --n-samples 64 --max-rows 45000000 --batch-size 16384
fi
echo "S2C_FINAL_DONE task=$SLURM_ARRAY_TASK_ID"; date
