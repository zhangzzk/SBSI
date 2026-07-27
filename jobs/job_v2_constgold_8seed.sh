#!/bin/bash
#SBATCH --job-name=v2cg8
#SBATCH --time=03:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%3
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/v2cg8_s%a_%j.out

# Gold-V2 (coupling-pinned lt500) 8-seed constgold certification + per-object dump.
# The dump (case, input_index, r_input_p, r_sim, R_flow, R_blend, neighbored, distance) is what
# figv2_fig2 (response vs galaxy properties) and figv2_fig3 (per-seed m) are both built from.
# constgold is EVAL-ONLY here -- the flow trained on the det_meas half-shear legs (firewall).
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
CAT=/project/ls-gruen/users/zekang.zhang/lsst_sims_fs2_25876_constant/constant_response_catalogue_train.feather
RES=/home/z/Zekang.Zhang/SBSI/results
DUMPDIR=/project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/v2_constgold_dumps
mkdir -p $DUMPDIR

SEEDS=(501 502 503 505 506 507 508 509)
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CK=$D/measurement_flow_g0_ngmix_ablate_s2c_coupling_lt500_s${S}_swaavg.pt

echo "### V2 CONSTGOLD seed=$S task=$SLURM_ARRAY_TASK_ID job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/validate_constant_with_blend.py \
  --measurement-model "$CK" --catalogue "$CAT" --min-case 40 \
  --blend-lookup "$RES/blend_lookup_extnbrho_c40-139.feather" \
  --crowd-flux-lookup "$RES/crowd_flux_conc_c0-199.feather" \
  --global-only --flow-seed 12345 --n-samples 64 --max-rows 45000000 --batch-size 16384 \
  --dump "$DUMPDIR/v2_perobj_s${S}.feather"
echo "V2CG8_DONE seed=$S"; date
