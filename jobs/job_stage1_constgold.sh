#!/bin/bash
#SBATCH --job-name=stage1_cg
#SBATCH --time=01:20:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64-v3
#SBATCH --partition=inter
#SBATCH --output=/home/z/Zekang.Zhang/logs/stage1_cg_%j.out

# Gold-V2 Stage-1 (SHAPE) validation on CONSTGOLD, step-by-step (owner 2026-07-23).
# Direction-A intent: NO empirical bridge (bridge=1.0). The reframe conditions on TRUE props and
# emits measured shape as an OUTPUT in the SAME ngmix estimator constgold's truth uses, so R_flow
# and R_sim are in the same units -> the bridge must be exactly 1.0. Any deviation is a MODEL error
# to diagnose, NOT a factor to absorb (cont.117: the isolated "win" was a bridge artifact).
# Clean gate = ISOLATED band (neighbored=False, R_blend=0): pure-flow m_iso = R_sim_iso/R_flow_iso-1.
# FIREWALL: constgold read for validation only; the flow trained on half-shear.
set -e
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd /home/z/Zekang.Zhang/SBSI

MODEL=${MODEL:-sw_th400}
CKPTGLOB=/project/ls-gruen/users/zekang.zhang/sbsi_caches/forward_proto/forward_${MODEL}_seed*_joint.pt
BLEND=results/blend_lookup_const_c0-40.feather
echo "### STAGE1_CG job=$SLURM_JOB_ID node=$SLURMD_NODENAME model=$MODEL ###"; nvidia-smi -L; date

# bridge=1.0 (NO empirical factor). ISOLATED band = pure flow test. BLENDED/ALL add R_blend (emulator,
# ngmix units, no bridge). R_blend caveats (cont.116 1.52x over-add) are secondary; the ISOLATED
# number is the Stage-1 headline.
python -B -u scripts/eval_constgold_closure.py \
  --ckpt-glob "$CKPTGLOB" \
  --bridge 1.0 \
  --blend-lookup "$BLEND" \
  --max-case 40 --true-re-min 0.3 --true-mag-max 26 \
  --output /project/ls-gruen/users/zekang.zhang/sbsi_caches/derisk/stage1_cg_${MODEL}.npz \
  || { echo "STAGE1_CG FAILED"; exit 1; }

echo "### STAGE1_CG_DONE job=$SLURM_JOB_ID model=$MODEL ###"; date
