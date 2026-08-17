#!/bin/bash
#SBATCH --job-name=sn_model
#SBATCH --time=00:45:00
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=inter
#SBATCH --constraint=x86-64-v3
#SBATCH --output=/home/z/Zekang.Zhang/logs/sn_model_%j.out

# Does the flow reproduce a PROXY-S/N cut (one it can actually represent)? Proxy on BOTH sides.
# The gate (15364971) showed the proxy overstates the REAL-S/N bias by 0.25-0.52 pts; that gap is
# separate from whatever m_flow this reports. Detection separated: both-detected pairs only.
eval "$(conda shell.bash hook)"; conda activate sims1
export PYTHONPATH="/home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot:/home/z/Zekang.Zhang/blendemu:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/z/Zekang.Zhang/SBSI/.claude/worktrees/selbias-plot
D=/project/ls-gruen/users/zekang.zhang/sbsi_caches/ablation
SEEDS="${SEEDS:-501 502 503 505}"
CK=""; for sd in $SEEDS; do CK="$CK $D/measurement_flow_g0_ngmix_ablate_s2c_lt500_dom6x6_s${sd}_swaavg.pt"; done
echo "### SN MODEL PREDICTION job=$SLURM_JOB_ID ###"; nvidia-smi -L; date
python -u scripts/eval_selection_sn_model.py --ckpt $CK --n-samples 32 --batch-size 16384 \
  2>&1 | grep -v "module command" || { echo SN_MODEL_FAILED; exit 1; }
echo SN_MODEL_ALL_DONE; date
