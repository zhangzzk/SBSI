#!/bin/bash
#SBATCH --job-name=pin_paired
#SBATCH --time=04:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/pin_paired_s%a_%j.out

# Paired pin-is-free test: for each seed, run constgold closure on BOTH the pre-pin baseline
# (s2_true4d) and the coupling-pinned (s2c_coupling_lt500) checkpoint. Same seed -> the seed
# noise cancels in the per-seed difference; the ensemble over 8 seeds tightens the m.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI-ablation:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI-ablation
SEEDS=(501 502 503 505 506 507 508 509)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
RES=/home/z/Zekang.Zhang/SBSI/results
run() {
  local ck=$1 tag=$2
  [ -f "$ck" ] || { echo "MISSING $tag $ck"; return; }
  echo "===== $tag seed=$SEED ====="
  python -u scripts/validate_constant_with_blend.py \
    --measurement-model "$ck" --catalogue "$CAT" --min-case 40 \
    --blend-lookup "$RES/blend_lookup_extnbrho_c40-139.feather" \
    --crowd-flux-lookup "$RES/crowd_flux_conc_c0-199.feather" \
    --global-only --flow-seed 12345 --n-samples 64 --max-rows 45000000 --batch-size 16384
}
echo "### PIN PAIRED CLOSURE seed=$SEED job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
run "$D/measurement_flow_g0_ngmix_ablate_s2_true4d_s${SEED}_swaavg.pt"        "BASELINE(no-pin)"
run "$D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt500_s${SEED}_swaavg.pt" "PINNED(lt500)"
echo "PIN_PAIRED_DONE seed=$SEED"; date
