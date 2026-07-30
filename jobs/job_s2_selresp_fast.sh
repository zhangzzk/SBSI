#!/bin/bash
#SBATCH --job-name=selresp_fast
#SBATCH --time=01:30:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/selresp_fast_%j.out

# Stage-2 selection-response gate on baseline V2 (dom6x6, 16 seeds) -- FAST variant.
# Same science as jobs/job_s2_selresp_v2base.sh; two changes, both about speed only:
#
#  1. NO `--all-too`. The ALL-objects scope has 2.3x the rows of ISOLATED and costs ~70% of the
#     total runtime, and the script itself labels it "diagnostic". The firewall-clean science
#     number is the ISOLATED one (no brighter true neighbour within 7" -> R_blend ~ 0, so no
#     emulator and no constgold enter). Dropping it changes NO reported science number.
#  2. `--tf32`. The flow is 10 coupling layers x 3 conditioner layers x hidden 256, i.e. almost pure
#     GEMM, so TF32 is the single biggest GPU lever. GATED on jobs/job_selresp_tf32_check.sh:
#     do NOT trust output from this job unless that gate printed "TF32 ADOPTED".
#
# NOT changed, deliberately: --n-samples 128. It sets the Monte Carlo noise on R_model, and the
# effects under study are percent-level, so buying speed there would spend the precision we need.
#
# NOT attempted here: the 16x redundant conditioning-frame rebuild in leg_draws (the frame depends
# only on (chunk, leg), not on the checkpoint, so a loop reorder would build it once instead of 16
# times), and keeping the draws on GPU instead of `.cpu().numpy()` per chunk (~205MB per leg-chunk
# transferred, then 8 cut masks evaluated in numpy on CPU). Both are real but are a refactor of
# model_selected_response with correctness risk; recorded so they are not forgotten.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot

D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
GLOB="$D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s*_swaavg.pt"
N=$(ls $GLOB 2>/dev/null | wc -l)
echo "### S2 SELECTION RESPONSE -- FAST (dom6x6, ${N} seeds, ISO only, TF32)  job=$SLURM_JOB_ID ###"
nvidia-smi -L; date
[ "$N" -eq 16 ] || echo "WARNING: expected 16 seeds, found $N"

python -u scripts/eval_selection_response.py \
  --ckpt-glob "$GLOB" --tf32 \
  --max-case 39 --n-samples 128 --batch-size 16384 \
  --output "$D/selection_response_v2base_dom6x6_16seed_fast.npz" \
  || { echo "SELRESP_FAST_FAILED"; exit 1; }
echo "SELRESP_FAST_DONE"; date
